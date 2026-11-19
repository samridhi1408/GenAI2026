"""Tests for the entry-task pipeline. No network or model access: the language
model is replaced by scripted handlers, one per role."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from bioloopbench.entrytasks import prompts
from bioloopbench.entrytasks.agent import run_episode
from bioloopbench.entrytasks.calibrate import calibrate_task
from bioloopbench.entrytasks.env import AskEnvironment, public_view
from bioloopbench.entrytasks.evaluate import evaluate, summarise
from bioloopbench.entrytasks.extract import check_drafts, extract_entries
from bioloopbench.entrytasks.judge import match_fixes, score_episode
from bioloopbench.entrytasks import pipeline
from bioloopbench.entrytasks.pipeline import ENTRY_DEFAULTS, protocol_path, stage_calibrate, stage_extract, write_manifest
from bioloopbench.taskgen import llm
from bioloopbench.taskgen.corpus import parse_article, protocol_text
from bioloopbench.taskgen.store import DEFAULTS, Funnel, read_json, write_json

XML = """<article><front><article-meta>
<article-id pub-id-type="pmcid">PMC1</article-id><article-id pub-id-type="doi">10.1/x1</article-id>
<title-group><article-title>Protocol for assembling plasmids</article-title></title-group>
<pub-date pub-type="epub"><day>3</day><month>8</month><year>2026</year></pub-date>
<pub-date pub-type="collection"><month>9</month><year>2026</year></pub-date>
<history><date date-type="received"><day>1</day><month>5</month><year>2026</year></date>
<date date-type="accepted"><day>20</day><month>7</month><year>2026</year></date></history>
<abstract><p>We describe plasmid assembly.</p></abstract></article-meta></front>
<body><sec><title>Step-by-step method details</title><list>
<list-item><label>3.</label><p>Digest the vector with BsaI for one hour at 37 degrees.</p></list-item>
<list-item><label>4.</label><p>Ligate with T4 ligase and transform chemically competent DH5alpha cells, then plate on ampicillin.</p></list-item>
</list></sec>
<sec><title>Troubleshooting</title>
<sec><title>Problem 1</title><p>No colonies appear on the plate after transformation (step 4).</p></sec>
<sec><title>Potential solution</title><p>The competent cells may have lost competence; use a fresh aliquot. Alternatively the ligase buffer ATP has degraded; use fresh buffer.</p></sec>
<sec><title>Problem 2</title><p>The sequencing provider reports a delay.</p></sec>
<sec><title>Potential solution</title><p>Wait for the provider.</p></sec>
</sec></body></article>"""

CONTEXT = ("A backbone was cut with a type IIS enzyme and joined to an insert using T4 ligase. The product was put "
           "into chemically competent DH5alpha and spread on selective agar.")
PROBLEM = "The selective plate was empty the next morning."
TASK1 = {"entry_index": 1, "technique": "plasmid assembly", "context": CONTEXT, "problem": PROBLEM,
         "problem_quote": "No colonies appear on the plate after transformation", "step_refs": ["4"],
         "fixes": [{"fix": "Repeat with a fresh aliquot of competent cells.", "fix_quote": "use a fresh aliquot",
                    "cause": "The cells have lost competence."},
                   {"fix": "Use fresh ligase buffer.", "fix_quote": "use fresh buffer", "cause": None}]}
SKIP2 = {"entry_index": 2, "reason_code": "not_observable"}
KEYWORDS = ("aliquot", "buffer")


def extraction(tasks=(TASK1,), skipped=(SKIP2,)):
    return {"tasks": [json.loads(json.dumps(t)) for t in tasks], "skipped": list(skipped)}


def generator(system, messages):
    p = messages[0]["content"]
    if p.startswith(prompts.PARAPHRASE[:40]):
        fixes = dict(re.findall(r"^(F\d+): (.+)$", p, re.M))
        return json.dumps({"paraphrases": {i: "Our suggestion: " + t for i, t in fixes.items()},
                           "vague": "Optimise the conditions."})
    return json.dumps(extraction())


def auditor(system, messages):
    return json.dumps({"tasks": [{"entry_index": 1, "bench_observable": True, "problem_leaks_fix": False,
                                  "context_unsupported": False, "identifies_source": False,
                                  "fixes": [{"supported": True, "concrete": True}, {"supported": True, "concrete": True}]}]})


def judge(system, messages):
    p = messages[0]["content"]
    fixes = re.findall(r"^(F\d+): (.+)$", p, re.M)
    out = []
    for label, text in re.findall(r"^(P\d+): (.+)$", p, re.M):
        hit = next((i for i, f in fixes for k in KEYWORDS if k in f.lower() and k in text.lower()), "none")
        out.append({"proposal": label, "matches": hit, "rationale": ""})
    return json.dumps({"matches": out})


def gatekeeper(system, messages):
    return json.dumps({"kind": "answer", "answer": "The cells were from an old box."})


def solver(fixes=("Use a fresh aliquot of cells.",), question="How old were the cells?", step="4"):
    """An agent that asks one question when it may, then answers."""
    def handler(system, messages):
        asked = any(m["role"] == "assistant" for m in messages)
        if question and not asked and "You may ask up to" in messages[0]["content"]:
            return json.dumps({"reasoning": "", "action": {"type": "ask", "question": question}})
        return json.dumps({"reasoning": "", "action": {"type": "answer", "fixes": list(fixes), "step": step}})
    return handler


class EntryTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.cfg = {**DEFAULTS, **ENTRY_DEFAULTS, "workers": 1, "models": {r: f"mock:{r}" for r in DEFAULTS["models"]}}
        llm.CACHE_DIR = None
        llm.MockBackend.handlers = {"generator": generator, "auditor": auditor, "gatekeeper": gatekeeper,
                                    "judge": judge, "solver": solver()}
        xml = self.run / "corpus" / "xml" / "PMC1.xml"
        xml.parent.mkdir(parents=True)
        xml.write_text(XML)
        self.article = parse_article(xml)
        self.protocol = protocol_text(self.article)

    def tearDown(self):
        llm.MockBackend.handlers = {}
        self.tmp.cleanup()

    def record(self):
        records, _ = extract_entries(self.article, self.cfg)
        return dict(records[0], split="test")

    # corpus ---------------------------------------------------------------
    def test_protocol_keeps_step_numbers_and_hides_troubleshooting(self):
        self.assertIn("4. Ligate with T4 ligase", self.protocol)
        self.assertNotIn("fresh aliquot", self.protocol)
        self.assertEqual([e["index"] for e in self.article["troubleshooting"]], [1, 2])

    # extraction -----------------------------------------------------------
    def test_valid_extraction_passes_checks(self):
        self.assertEqual(check_drafts(extraction(), self.article, 9), [])

    def test_every_entry_must_be_accounted_for(self):
        self.assertTrue(any("exactly once" in p for p in check_drafts(extraction(skipped=()), self.article, 9)))

    def test_fabricated_quotes_are_rejected(self):
        bad = extraction()
        bad["tasks"][0]["fixes"][0]["fix_quote"] = "replace the incubator"
        bad["tasks"][0]["problem_quote"] = "the plate is contaminated"
        problems = check_drafts(bad, self.article, 9)
        self.assertTrue(any("fix_quote" in p for p in problems))
        self.assertTrue(any("problem_quote" in p for p in problems))

    def test_step_reference_must_appear_in_entry(self):
        bad = extraction()
        bad["tasks"][0]["step_refs"] = ["7"]
        self.assertTrue(any("step reference" in p for p in check_drafts(bad, self.article, 9)))

    def test_context_copied_from_article_is_rejected(self):
        bad = extraction()
        bad["tasks"][0]["context"] = "Ligate with T4 ligase and transform chemically competent DH5alpha cells, then plate on ampicillin."
        self.assertTrue(any("copies" in p for p in check_drafts(bad, self.article, 9)))

    def test_unknown_skip_code_and_missing_fix_are_rejected(self):
        bad = extraction(skipped=({"entry_index": 2, "reason_code": "boring"},))
        bad["tasks"][0]["fixes"] = []
        problems = check_drafts(bad, self.article, 9)
        self.assertTrue(any("reason_code" in p for p in problems))
        self.assertTrue(any("at least one fix" in p for p in problems))

    def test_extract_builds_record(self):
        records, skipped = extract_entries(self.article, self.cfg)
        self.assertEqual(skipped, {"not_observable": 1})
        r = records[0]
        self.assertEqual((r["task_id"], r["status"], r["reasons"]), ("PMC1_e1", "extracted", []))
        self.assertEqual([f["id"] for f in r["fixes"]], ["F1", "F2"])
        self.assertIn(PROBLEM, r["opening"])
        self.assertEqual(r["budget"], self.cfg["question_budget"])

    def test_generator_that_never_grounds_its_quotes_fails_the_article(self):
        bad = extraction()
        bad["tasks"][0]["fixes"][0]["fix_quote"] = "not in the entry"
        llm.MockBackend.handlers["generator"] = lambda s, m: json.dumps(bad)
        with self.assertRaises(llm.LLMError):
            extract_entries(self.article, self.cfg)

    def test_auditor_drops_unsupported_fix_and_renumbers(self):
        def partial(system, messages):
            a = json.loads(auditor(system, messages))
            a["tasks"][0]["fixes"][0]["supported"] = False
            return json.dumps(a)
        llm.MockBackend.handlers["auditor"] = partial
        r = extract_entries(self.article, self.cfg)[0][0]
        self.assertEqual(r["status"], "extracted")
        self.assertEqual([(f["id"], f["fix"]) for f in r["fixes"]], [("F1", "Use fresh ligase buffer.")])
        self.assertEqual(r["dropped_fixes"], 1)

    def test_auditor_flags_reject_the_task(self):
        def strict(system, messages):
            a = json.loads(auditor(system, messages))
            a["tasks"][0].update(problem_leaks_fix=True, bench_observable=False,
                                 fixes=[{"supported": False, "concrete": True}, {"supported": True, "concrete": False}])
            return json.dumps(a)
        llm.MockBackend.handlers["auditor"] = strict
        r = extract_entries(self.article, self.cfg)[0][0]
        self.assertEqual(r["status"], "rejected")
        self.assertEqual(set(r["reasons"]), {"not_bench_observable", "problem_leaks_fix", "no_supported_fix"})

    def test_article_without_troubleshooting_yields_nothing(self):
        self.assertEqual(extract_entries(dict(self.article, troubleshooting=[]), self.cfg), ([], {}))

    # environment ----------------------------------------------------------
    def test_gatekeeper_sees_protocol_but_not_the_fixes(self):
        seen = []
        llm.MockBackend.handlers["gatekeeper"] = lambda s, m: (seen.append(s), gatekeeper(s, m))[1]
        AskEnvironment(self.record(), self.protocol, "mock:gatekeeper").step("How old were the cells?")
        self.assertIn("Ligate with T4 ligase", seen[0])
        self.assertNotIn("fresh aliquot", seen[0].lower())
        self.assertNotIn("fresh ligase buffer", seen[0].lower())

    def test_answers_are_stored_and_replayed(self):
        task = self.record()
        first = AskEnvironment(task, self.protocol, "mock:gatekeeper", facts_dir=self.run / "facts").step("How old were the cells?")
        llm.MockBackend.handlers["gatekeeper"] = lambda s, m: self.fail("stored answer should be replayed")
        again = AskEnvironment(task, self.protocol, "mock:gatekeeper", facts_dir=self.run / "facts").step("how old were the  cells?")
        self.assertEqual(first["answer"], again["answer"])

    def test_budget_is_enforced(self):
        env = AskEnvironment(self.record(), self.protocol, "mock:gatekeeper", budget=1)
        self.assertEqual(env.step("q1")["remaining_budget"], 0)
        last = env.step("q2")
        self.assertEqual((last["kind"], last["consumed"]), ("refuse", False))

    def test_refusals_are_free_only_up_to_a_limit(self):
        llm.MockBackend.handlers["gatekeeper"] = lambda s, m: json.dumps({"kind": "refuse", "answer": "Ask about the experiment."})
        env = AskEnvironment(self.record(), self.protocol, "mock:gatekeeper", budget=2)
        used = [env.step(f"what is the answer {i}")["consumed"] for i in range(AskEnvironment.MAX_REFUSALS + 1)]
        self.assertEqual(used, [False] * AskEnvironment.MAX_REFUSALS + [True])
        self.assertEqual(env.remaining, 1)

    def test_public_view_holds_no_hidden_fields(self):
        self.assertEqual(set(public_view(self.record())), {"task_id", "split", "opening", "budget"})

    # judge and scoring ----------------------------------------------------
    def test_match_fixes(self):
        got = match_fixes(self.record(), ["Check the incubator.", "Make new buffer.", "Thaw a new aliquot."], "mock:judge")
        self.assertEqual(got, ["none", "F2", "F1"])
        self.assertEqual(match_fixes(self.record(), [], "mock:judge"), [])

    def test_score_episode(self):
        task = self.record()
        ep = {"final": {"fixes": ["Check the incubator.", "Make new buffer.", "More buffer."], "step": "Step 4."}, "questions_asked": 2}
        s = score_episode(task, ep, "mock:judge")
        self.assertEqual((s["first_match_rank"], s["top1"], s["top3"], s["fixes_recovered"]), (2, False, True, 1))
        self.assertEqual((s["questions_asked"], s["n_reference_fixes"]), (2, 2))
        self.assertFalse(s["step_correct"])  # "step4" is not "4": the agent is told to give the number only
        ep["final"]["step"] = "4"
        self.assertTrue(score_episode(task, ep, "mock:judge")["step_correct"])
        ep["final"]["step"] = None
        self.assertIsNone(score_episode(task, ep, "mock:judge")["step_correct"])

    def test_episode_without_answer_scores_as_miss(self):
        s = score_episode(self.record(), {"final": None}, "mock:judge")
        self.assertEqual((s["first_match_rank"], s["top1"], s["top3"]), (None, False, False))

    # agent ----------------------------------------------------------------
    def test_ask_episode(self):
        ep = run_episode(self.record(), self.protocol, "mock:solver", "mock:gatekeeper", mode="ask")
        self.assertEqual((ep["status"], ep["questions_asked"], len(ep["rounds"])), ("completed", 1, 2))
        self.assertEqual(ep["rounds"][0]["result"]["kind"], "answer")
        self.assertIsNone(ep["final"]["step"])  # the step is only asked for with the full protocol

    def test_symptom_only_cannot_ask_or_see_protocol(self):
        seen = []
        inner = solver()
        llm.MockBackend.handlers["solver"] = lambda s, m: (seen.append(m[0]["content"]), inner(s, m))[1]
        ep = run_episode(self.record(), self.protocol, "mock:solver", "mock:gatekeeper", mode="symptom_only")
        self.assertEqual((ep["status"], ep["questions_asked"], len(ep["rounds"])), ("completed", 0, 1))
        self.assertNotIn("Ligate with T4 ligase", seen[0])

    def test_full_protocol_mode_shows_protocol_and_records_step(self):
        seen = []
        inner = solver()
        llm.MockBackend.handlers["solver"] = lambda s, m: (seen.append(m[0]["content"]), inner(s, m))[1]
        ep = run_episode(self.record(), self.protocol, "mock:solver", "mock:gatekeeper", mode="full_protocol")
        self.assertIn("Ligate with T4 ligase", seen[0])
        self.assertEqual((ep["questions_asked"], ep["final"]["step"]), (0, "4"))

    def test_agent_that_keeps_asking_when_it_cannot_is_an_error(self):
        llm.MockBackend.handlers["solver"] = lambda s, m: json.dumps({"reasoning": "", "action": {"type": "ask", "question": "why?"}})
        ep = run_episode(self.record(), self.protocol, "mock:solver", "mock:gatekeeper", mode="symptom_only")
        self.assertEqual(ep["status"], "error")
        self.assertIsNone(ep["final"])

    def test_overlong_or_too_many_fixes_are_refused(self):
        llm.MockBackend.handlers["solver"] = solver(fixes=["a", "b", "c", "d"])
        self.assertEqual(run_episode(self.record(), self.protocol, "mock:solver", "mock:gatekeeper", mode="symptom_only")["status"], "error")
        llm.MockBackend.handlers["solver"] = solver(fixes=["word " * 80])
        self.assertEqual(run_episode(self.record(), self.protocol, "mock:solver", "mock:gatekeeper", mode="symptom_only")["status"], "error")

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            run_episode(self.record(), self.protocol, "mock:solver", "mock:gatekeeper", mode="guess")

    # calibration ----------------------------------------------------------
    def test_calibration_accepts_and_flags_symptom_solved(self):
        t = calibrate_task(self.record(), self.protocol, self.cfg)
        self.assertEqual((t["status"], t["reasons"], t["symptom_solved"]), ("accepted", [], True))
        self.assertTrue(t["calibration"]["judge_sanity"]["passed"])
        self.assertEqual(t["calibration"]["symptom_only"]["top1_hits"], self.cfg["symptom_only_samples"])

    def test_task_not_solved_from_symptom_is_not_flagged(self):
        llm.MockBackend.handlers["solver"] = solver(fixes=["Check the incubator."])
        t = calibrate_task(self.record(), self.protocol, self.cfg)
        self.assertEqual((t["status"], t["symptom_solved"]), ("accepted", False))
        self.assertEqual(t["calibration"]["symptom_only"]["top3_hits"], 0)

    def test_judge_that_matches_vague_advice_rejects_task(self):
        def lenient(system, messages):
            labels = re.findall(r"^(P\d+): ", messages[0]["content"], re.M)
            return json.dumps({"matches": [{"proposal": p, "matches": "F1", "rationale": ""} for p in labels]})
        llm.MockBackend.handlers["judge"] = lenient
        t = calibrate_task(self.record(), self.protocol, self.cfg)
        self.assertEqual((t["status"], t["reasons"]), ("rejected", ["judge_cannot_score"]))

    # pipeline stages and evaluation ---------------------------------------
    def run_stages(self):
        topic = {"id": "cloning"}
        write_json(self.run / "topics" / "cloning" / "articles.json",
                   {"articles": [{"pmcid": "PMC1", "split": "test"}, {"pmcid": "PMC9", "split": "test", "error": "x"}]})
        funnel = Funnel(self.run / "funnel.json")
        stage_extract(self.run, topic, self.cfg, funnel)
        stage_calibrate(self.run, topic, self.cfg, funnel)
        return funnel.data["cloning"]

    def test_stages_write_records_and_funnel(self):
        f = self.run_stages()
        self.assertEqual(f["extract"], {"articles_read": 1, "entries_seen": 2, "skipped:not_observable": 1,
                                        "tasks_extracted_test": 1, "fixes_kept": 2, "tasks_with_step_ref": 1})
        self.assertEqual(f["calibrate"], {"tasks_accepted_test": 1, "symptom_solved_test": 1})
        self.assertEqual(read_json(self.run / "topics" / "cloning" / "entries.json"), ["PMC1_e1"])
        self.assertEqual(protocol_path(self.run, "PMC1").read_text(), self.protocol)
        public = read_json(self.run / "public" / "PMC1_e1.json")
        self.assertEqual(set(public), {"task_id", "split", "opening", "budget"})
        self.assertEqual(read_json(self.run / "private" / "PMC1_e1.json")["status"], "accepted")

    def test_article_dates_are_parsed_as_written(self):
        a = self.article
        self.assertEqual(a["date"], "2026-08-03")
        self.assertEqual(a["pub_dates"], {"epub": "2026-08-03", "collection": "2026-09"})
        self.assertEqual(a["history_dates"], {"received": "2026-05-01", "accepted": "2026-07-20"})

    def corpus(self, topic, hits, **cfg):
        real = pipeline.search
        pipeline.search = lambda phrases, limit=None: [dict(h) for h in hits]
        try:
            funnel = Funnel(self.run / "funnel.json")
            pipeline.stage_corpus(self.run, {"id": topic, "search_phrases": ["x"]}, {**self.cfg, **cfg}, funnel)
        finally:
            pipeline.search = real
        return funnel.data[topic]["corpus"], read_json(self.run / "topics" / topic / "articles.json")["articles"]

    def test_corpus_takes_test_articles_first_and_skips_other_topics_articles(self):
        xml = self.run / "corpus" / "xml"
        for n, year in ((2, 2024), (3, 2023), (4, 2022)):
            (xml / f"PMC{n}.xml").write_text(XML.replace("PMC1", f"PMC{n}").replace("2026", str(year)))
        hit = lambda n, d: {"pmcid": f"PMC{n}", "date": d, "title": "", "pmid": str(n), "doi": "", "pub_year": d[:4],
                            "first_publication_date": d, "first_index_date": "2026-08-05", "journal": "STAR Protocols"}
        hits = [hit(1, "2026-08-03"), hit(2, "2024-08-03"), hit(3, "2023-08-03"), hit(4, "2022-08-03")]
        f, arts = self.corpus("a", hits, articles_per_topic=2)
        self.assertEqual([(a["pmcid"], a["split"]) for a in arts if a["split"] == "test"], [("PMC1", "test")])
        self.assertEqual((len(arts), f["test_articles_selected"], f["development_articles_selected"]), (2, 1, 1))
        self.assertEqual(arts[0]["pub_dates"]["epub"][5:], "08-03")
        self.assertEqual(arts[0]["first_index_date"], "2026-08-05")
        f2, arts2 = self.corpus("b", hits, articles_per_topic=4)
        self.assertEqual(f2["already_in_another_topic"], 2)
        self.assertEqual(len(arts2), 2)
        self.assertFalse({a["pmcid"] for a in arts} & {a["pmcid"] for a in arts2})

    def test_records_and_manifest_carry_dates(self):
        f, _ = self.corpus("cloning", [{"pmcid": "PMC1", "date": "2026-08-03", "title": "", "pmid": "77", "doi": "10.1/x1",
                                        "pub_year": "2026", "first_publication_date": "2026-08-03",
                                        "first_index_date": "2026-08-05", "journal": "STAR Protocols"}])
        funnel = Funnel(self.run / "funnel.json")
        stage_extract(self.run, {"id": "cloning"}, self.cfg, funnel)
        stage_calibrate(self.run, {"id": "cloning"}, self.cfg, funnel)
        t = read_json(self.run / "private" / "PMC1_e1.json")
        self.assertEqual((t["date"], t["split"], t["article"]["pmid"]), ("2026-08-03", "test", "77"))
        self.assertEqual(t["article"]["history_dates"]["accepted"], "2026-07-20")
        self.assertTrue(t["extracted_at"])
        self.assertEqual(write_manifest(self.run), 1)
        import csv
        row = next(csv.DictReader(open(self.run / "manifest.csv")))
        self.assertEqual((row["task_id"], row["status"], row["date"], row["pub_date_collection"], row["date_received"],
                          row["first_index_date"], row["pmid"], row["n_fixes"], row["symptom_solved"]),
                         ("PMC1_e1", "accepted", "2026-08-03", "2026-09", "2026-05-01", "2026-08-05", "77", "2", "True"))
        self.assertEqual(json.loads((self.run / "manifest.jsonl").read_text())["split"], "test")
        self.assertEqual(read_json(self.run / "progress.json")["done"], 1)

    def test_recalibration_withdraws_a_task_the_judge_can_no_longer_score(self):
        self.run_stages()
        llm.MockBackend.handlers["judge"] = lambda s, m: json.dumps({"matches": [
            {"proposal": p, "matches": "none", "rationale": ""} for p in re.findall(r"^(P\d+): ", m[0]["content"], re.M)]})
        funnel = Funnel(self.run / "funnel.json")
        stage_calibrate(self.run, {"id": "cloning"}, self.cfg, funnel)
        self.assertEqual(funnel.data["cloning"]["calibrate"], {"tasks_rejected_test": 1, "reject:judge_cannot_score": 1})
        self.assertFalse((self.run / "public" / "PMC1_e1.json").exists())

    def test_evaluate_gives_agent_only_the_public_view(self):
        self.run_stages()
        s = evaluate(self.run, "mock:solver", ["ask", "full_protocol", "symptom_only"], "test", self.cfg)
        ask = s["modes"]["ask"]
        self.assertEqual((ask["n"], ask["completed"], ask["top1"], ask["mean_questions_asked"]), (1, 1, 1.0, 1.0))
        self.assertIsNone(ask["step_accuracy"])
        self.assertEqual((s["modes"]["full_protocol"]["step_accuracy"], s["modes"]["full_protocol"]["n_step_scored"]), (1.0, 1))
        self.assertEqual((ask["n_excluding_symptom_solved"], ask["top1_excluding_symptom_solved"]), (0, None))
        out = self.run / "results" / "mock_solver"
        self.assertTrue((out / "summary.json").exists())
        self.assertEqual(read_json(out / "ask.json")[0]["episode"]["status"], "completed")
        self.assertEqual(evaluate(self.run, "mock:solver", ["ask"], "development", self.cfg)["modes"]["ask"], {"n": 0})

    def test_summarise_counts_failures_as_misses(self):
        rows = [{"top1": True, "top3": True, "questions_asked": 2, "step_correct": True, "status": "completed", "symptom_solved": True},
                {"top1": False, "top3": False, "questions_asked": 0, "step_correct": None, "status": "error", "symptom_solved": False}]
        s = summarise(rows)
        self.assertEqual((s["n"], s["completed"], s["top1"], s["top3"], s["mean_questions_asked"]), (2, 1, 0.5, 0.5, 1.0))
        self.assertEqual((s["top1_excluding_symptom_solved"], s["n_excluding_symptom_solved"], s["step_accuracy"]), (0.0, 1, 1.0))


if __name__ == "__main__":
    unittest.main()
