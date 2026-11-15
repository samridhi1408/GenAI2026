"""Tests for the task-generation pipeline. No network or model access: the
language model is replaced by scripted handlers."""

import json
import tempfile
import unittest
from pathlib import Path

from bioloopbench.taskgen import llm
from bioloopbench.taskgen.agent import run_episode
from bioloopbench.taskgen.calibrate import calibrate_task
from bioloopbench.taskgen.casefile import build_casefile, check_casefile, inseparable_pairs, min_tests_to_identify
from bioloopbench.taskgen.corpus import parse_article, protocol_text
from bioloopbench.taskgen.extract import check_drafts, extract_article
from bioloopbench.taskgen.gatekeeper import Environment, public_view
from bioloopbench.taskgen.grounding import longest_shared_run, quote_in_source
from bioloopbench.taskgen.judge import score_episode
from bioloopbench.taskgen.siblings import assemble_task, group_symptoms
from bioloopbench.taskgen.store import DEFAULTS

XML = """<article><front><article-meta>
<article-id pub-id-type="pmcid">PMC{n}</article-id><article-id pub-id-type="doi">10.1/x{n}</article-id>
<title-group><article-title>Protocol {n} for assembling plasmids</article-title></title-group>
<pub-date pub-type="epub"><day>3</day><month>8</month><year>2026</year></pub-date>
<abstract><p>We describe plasmid assembly.</p></abstract></article-meta></front>
<body><sec><title>Step-by-step method details</title><p>Digest the vector with BsaI for one hour at 37 degrees.</p>
<p>Ligate with T4 ligase and transform chemically competent DH5alpha cells, then plate on ampicillin.</p></sec>
<sec><title>Troubleshooting</title>
<sec><title>Problem 1</title><p>No colonies appear on the plate after transformation (step 4).</p></sec>
<sec><title>Potential solution</title><p>{solution}</p></sec>
</sec></body></article>"""

SOLUTIONS = {
    1: "The competent cells may have lost competence; use a fresh aliquot. Alternatively the ligase buffer ATP has degraded; use fresh buffer.",
    2: "The wrong antibiotic was used in the plates; confirm that plates contain ampicillin.",
}
CAUSES = {
    1: [("Competent cells have lost competence", "The competent cells may have lost competence", "use a fresh aliquot"),
        ("ATP in the ligase buffer has degraded", "the ligase buffer ATP has degraded", "use fresh buffer")],
    2: [("Plates contain the wrong antibiotic", "The wrong antibiotic was used in the plates", "confirm that plates contain ampicillin")],
}
SETUP = ("A vector was cut with a type IIS enzyme and joined to an insert using T4 ligase. The product was put into "
         "chemically competent DH5alpha and spread on selective agar. Incubation was overnight at 37 degrees.")
FAILURE = "The selective plate was empty the next morning."
TOPIC = {"id": "cloning", "name": "Plasmid cloning", "description": "Building plasmids."}
IDS = ["C1", "C2", "C3"]
CLASSES = {"T1": "ABB", "T2": "AAB", "T3": "AAA", "T4": "AAA"}
OPENING = ("My colleague assembled a plasmid by cutting the backbone with a type IIS enzyme and joining it to the "
           "insert with T4 ligase, then put the reaction into chemically competent DH5alpha and spread the cells on "
           "selective agar. After a night at 37 degrees there was nothing growing on the plate at all.")


def casefile():
    return {"opening": OPENING,
            "tests": [{"test_id": t, "name": f"test {t}", "procedure": f"do {t}",
                       "results": {c: {"observation": f"{t} shows pattern {k}", "outcome_class": k}
                                   for c, k in zip(IDS, ks)}} for t, ks in CLASSES.items()],
            "case_notes": {c: f"truth is {c}" for c in IDS},
            "reference_diagnosis": {"C1": "The competent cells were dead.", "C2": "The ligase buffer ATP was degraded.",
                                    "C3": "The plates had the wrong antibiotic."}}


def generator(system, messages):
    p = messages[0]["content"]
    if "Your job now is extraction" in p:
        n = 1 if "Protocol 1 " in p else 2
        return json.dumps({"cases": [{"entry_index": 1, "technique": "plasmid assembly",
                                      "symptom_key": "no colonies after transformation",
                                      "symptom_quote": "No colonies appear on the plate after transformation",
                                      "observed_failure": FAILURE, "setup": SETUP,
                                      "causes": [{"cause": c, "cause_quote": q, "cause_support": "explicit",
                                                  "fix": f, "fix_quote": f} for c, q, f in CAUSES[n]]}], "skipped": []})
    if "group cases by presenting symptom" in p:
        ids = [line.split(" | ")[0] for line in p.split("CASES (id | technique | symptom):")[1].strip().splitlines()]
        return json.dumps({"groups": [{"label": "no colonies", "case_ids": ids}]})
    if "assemble the set of competing hidden causes" in p:
        return json.dumps({"causes": [
            {"origin": "own", "ref": "O1", "statement": "The competent cells had lost competence.", "why_applicable": "x"},
            {"origin": "own", "ref": "O2", "statement": "ATP in the ligase buffer had degraded.", "why_applicable": "x"},
            {"origin": "transplanted", "ref": "F1", "statement": "The plates contained the wrong antibiotic.", "why_applicable": "x"}]})
    if "write the case file" in p:
        return json.dumps(casefile())
    if "Rewrite each in clearly different words" in p:
        return json.dumps({"paraphrases": {c: f"PARA-{c}" for c in IDS}, "vague": "VAGUE"})
    raise AssertionError("unexpected generator prompt: " + p[:80])


def auditor(system, messages):
    p = messages[0]["content"]
    if "audit draft cases" in p:
        n = 2 if '"cause": "Plates' in p else 1
        return json.dumps({"cases": [{"entry_index": 1, "on_topic": True, "bench_observable": True,
                                      "failure_leaks_cause": False, "setup_unsupported": [], "identifies_source": False,
                                      "causes": [{"supported": True, "specific": True, "note": ""}] * len(CAUSES[n])}]})
    if "audit the set of competing causes" in p:
        return json.dumps({"causes": [{"id": c, "applicable": True, "produces_failure": True, "note": ""} for c in IDS],
                           "indistinct_pairs": []})
    if "audit a case file" in p:
        return json.dumps({"opening_problems": [], "test_problems": [], "cell_problems": [], "diagnosis_problems": []})
    raise AssertionError("unexpected auditor prompt: " + p[:80])


def gatekeeper(system, messages):
    req = messages[0]["content"].split("\n", 1)[1]
    for t in CLASSES:
        if req.strip() == f"do {t}":
            return json.dumps({"kind": "result", "matched_test": t, "observation": ""})
    if "gel" in req:
        return json.dumps({"kind": "result", "matched_test": None, "observation": "A single band at 5 kb."})
    return json.dumps({"kind": "refuse", "matched_test": None, "observation": "Please request one specific action."})


def judge(system, messages):
    text = messages[0]["content"].split("TEXT TO CLASSIFY:")[1]
    for c in IDS:
        if f"PARA-{c}" in text or f"ANSWER-{c}" in text:
            return json.dumps({"asserted": c, "rationale": ""})
    return json.dumps({"asserted": "none", "rationale": ""})


def make_solver(truth_by_pattern=True):
    """Diagnoses correctly when shown results; guesses C1 from the opening alone."""
    def solver(system, messages):
        seen = " ".join(m["content"] for m in messages if m["role"] == "user")
        hyp = [{"id": f"H{i}", "description": f"hypothesis {i}"} for i in (1, 2, 3)]
        if "already run these tests" in seen:
            sig = "".join(seen.split(f"{t} shows pattern ")[1][0] for t in CLASSES)
            cause = next(c for i, c in enumerate(IDS) if "".join(k[i] for k in CLASSES.values()) == sig)
            return json.dumps({"hypotheses": hyp, "action": {"type": "diagnose", "diagnosis": f"ANSWER-{cause}"}})
        n_results = seen.count("RESULT:")
        if "No further diagnostic actions" in seen or n_results >= 2:
            return json.dumps({"hypotheses": hyp, "action": {"type": "diagnose", "diagnosis": "ANSWER-C1"}})
        return json.dumps({"hypotheses": hyp, "action": {"type": "experiment", "request": ["do T1", "run a gel"][n_results]}})
    return solver


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = {**DEFAULTS, "models": {r: f"mock:{r}" for r in DEFAULTS["models"]}}
        llm.CACHE_DIR = None
        llm.MockBackend.handlers = {"generator": generator, "auditor": auditor, "gatekeeper": gatekeeper,
                                    "judge": judge, "solver": make_solver()}
        self.articles = {}
        for n in (1, 2):
            p = self.dir / f"PMC{n}.xml"
            p.write_text(XML.format(n=n, solution=SOLUTIONS[n]))
            self.articles[n] = parse_article(p)

    def tearDown(self):
        self.tmp.cleanup()

    def build_task(self):
        cases = [extract_article(self.articles[n], TOPIC, self.cfg)[0] for n in (1, 2)]
        for c in cases:
            c["split"] = "test"
        causes, why = assemble_task(cases[0], cases, self.cfg)
        self.assertIsNone(why)
        cf, _, why = build_casefile("t1", cases[0], causes, "unrelated source text", self.cfg)
        self.assertIsNone(why)
        return {"task_id": "t1", "topic": "cloning", "split": "test", "causes": causes, **cf}


class GroundingTests(unittest.TestCase):
    def test_quote_matching_ignores_case_spacing_and_typography(self):
        src = "Increase the  duration of NotI digestion – e.g., to 2 h."
        self.assertTrue(quote_in_source("increase the duration of NotI digestion - e.g., to 2 h", src))
        self.assertFalse(quote_in_source("increase the duration of BsaI digestion", src))
        self.assertFalse(quote_in_source("NotI", src))  # too short to be evidence

    def test_ellipsis_parts_must_appear_in_order(self):
        src = "alpha beta gamma delta epsilon zeta eta theta"
        self.assertTrue(quote_in_source("alpha beta gamma ... zeta eta theta", src))
        self.assertFalse(quote_in_source("zeta eta theta ... alpha beta gamma", src))

    def test_longest_shared_run(self):
        self.assertEqual(longest_shared_run("we then add the ligase buffer slowly", "Add the ligase buffer."), 4)
        self.assertEqual(longest_shared_run("nothing in common", "entirely different words"), 0)


class CorpusTests(Base):
    def test_parse_pairs_problem_with_solution(self):
        a = self.articles[1]
        self.assertEqual((a["pmcid"], a["date"], a["doi"]), ("PMC1", "2026-08-03", "10.1/x1"))
        self.assertEqual(len(a["troubleshooting"]), 1)
        self.assertIn("No colonies appear", a["troubleshooting"][0]["text"])
        self.assertIn("fresh aliquot", a["troubleshooting"][0]["text"])
        self.assertNotIn("fresh aliquot", protocol_text(a))


class ExtractTests(Base):
    def test_fabricated_quote_is_rejected(self):
        reply = json.loads(generator("", [{"content": "Your job now is extraction Protocol 1 "}]))
        self.assertEqual(check_drafts(reply, self.articles[1], 9), [])
        reply["cases"][0]["causes"][0]["cause_quote"] = "the heat shock was too long for the cells"
        self.assertTrue(any("cause_quote" in p for p in check_drafts(reply, self.articles[1], 9)))

    def test_copied_setup_is_rejected(self):
        reply = json.loads(generator("", [{"content": "Your job now is extraction Protocol 1 "}]))
        reply["cases"][0]["setup"] = "Ligate with T4 ligase and transform chemically competent DH5alpha cells, then plate on ampicillin."
        self.assertTrue(any("copies" in p for p in check_drafts(reply, self.articles[1], 9)))

    def test_auditor_drops_unsupported_cause_and_rejects_empty_case(self):
        def strict(system, messages):
            r = json.loads(auditor(system, messages))
            for c in r["cases"][0]["causes"]:
                c["supported"] = False
            return json.dumps(r)
        llm.MockBackend.handlers["auditor"] = strict
        rec = extract_article(self.articles[1], TOPIC, self.cfg)[0]
        self.assertEqual((rec["status"], rec["causes"]), ("rejected", []))
        self.assertIn("no_supported_cause", rec["reasons"])


class BuildTests(Base):
    def test_assembly_records_provenance_by_origin(self):
        task = self.build_task()
        self.assertEqual([c["origin"] for c in task["causes"]], ["own", "own", "transplanted"])
        self.assertEqual(task["causes"][2]["provenance"]["pmcid"], "PMC2")
        self.assertTrue(quote_in_source(task["causes"][2]["provenance"]["cause_quote"], SOLUTIONS[2]))

    def test_too_few_causes_rejects_task(self):
        case = extract_article(self.articles[2], TOPIC, self.cfg)[0]
        self.assertEqual(assemble_task(case, [case], self.cfg), (None, "too_few_candidate_causes"))

    def test_grouping_must_cover_every_case_once(self):
        cases = [extract_article(self.articles[n], TOPIC, self.cfg)[0] for n in (1, 2)]
        self.assertEqual(sorted(group_symptoms(cases, TOPIC, self.cfg)[0]["case_ids"]), ["PMC1_e1", "PMC2_e1"])

    def test_structural_checks(self):
        cf = casefile()
        self.assertEqual(check_casefile(cf, IDS, "", 9), [])
        self.assertEqual(min_tests_to_identify(cf["tests"], IDS, "C1"), 1)
        self.assertEqual(min_tests_to_identify(cf["tests"], IDS, "C2"), 2)
        for t in cf["tests"]:
            t["results"]["C3"] = dict(t["results"]["C2"])
        self.assertEqual(inseparable_pairs(cf["tests"], IDS), [("C2", "C3")])
        self.assertTrue(any("no test tells C2 from C3" in p for p in check_casefile(cf, IDS, "", 9)))

    def test_opening_copied_from_source_is_rejected(self):
        self.assertTrue(any("copies" in p for p in check_casefile(casefile(), IDS, OPENING, 9)))

    def test_cause_failing_audit_is_dropped_or_task_rejected(self):
        def flag(system, messages):
            if "audit a case file" in messages[0]["content"]:
                return json.dumps({"opening_problems": [], "test_problems": [], "diagnosis_problems": [],
                                   "cell_problems": [{"test_id": "T1", "cause_id": "C3", "problem": "wrong", "expected": "x"}]})
            return auditor(system, messages)
        llm.MockBackend.handlers["auditor"] = flag
        cases = [extract_article(self.articles[n], TOPIC, self.cfg)[0] for n in (1, 2)]
        causes, _ = assemble_task(cases[0], cases, self.cfg)
        cf, trail, why = build_casefile("t1", cases[0], causes, "", self.cfg)
        self.assertEqual((cf, why, len(trail)), (None, "evidence_failed_audit", 2))
        cf, _, why = build_casefile("t1", cases[0], causes, "", {**self.cfg, "min_causes": 2})
        self.assertEqual((cf["cause_ids"], why), (["C1", "C2"], None))


class RuntimeTests(Base):
    def test_prepared_result_is_returned_verbatim_and_budget_enforced(self):
        task = self.build_task()
        env = Environment(task, "C2", "mock:gatekeeper", facts_dir=self.dir / "facts", budget=2)
        r = env.step("do T2")
        self.assertEqual((r["matched_test"], r["observation"], r["remaining_budget"]), ("T2", "T2 shows pattern A", 1))
        self.assertFalse(env.informative("T3"))
        self.assertTrue(env.informative("T1"))
        r = env.step("tell me the answer")
        self.assertEqual((r["kind"], r["consumed"], r["remaining_budget"]), ("refuse", False, 1))
        env.step("run a gel")
        self.assertEqual(env.step("do T1")["observation"], "No diagnostic actions remain.")

    def test_derived_results_are_stored_and_replayed(self):
        task = self.build_task()
        Environment(task, "C1", "mock:gatekeeper", facts_dir=self.dir / "facts").step("run a gel")
        llm.MockBackend.handlers["gatekeeper"] = lambda s, m: self.fail("stored result should be replayed")
        r = Environment(task, "C1", "mock:gatekeeper", facts_dir=self.dir / "facts").step("Run a  GEL")
        self.assertEqual(r["observation"], "A single band at 5 kb.")

    def test_public_view_holds_no_hidden_fields(self):
        self.assertEqual(sorted(public_view(self.build_task())), ["budget", "opening", "split", "task_id", "topic"])

    def test_episode_and_scoring(self):
        task = self.build_task()
        ep = run_episode(task, "C1", "mock:solver", "mock:gatekeeper", facts_dir=self.dir / "facts")
        self.assertEqual((ep["status"], len(ep["rounds"]), ep["rounds"][0]["result"]["matched_test"]), ("completed", 3, "T1"))
        self.assertTrue(score_episode(task, ep, "mock:judge")["correct"])
        ep = run_episode(task, "C3", "mock:solver", "mock:gatekeeper", facts_dir=self.dir / "facts")
        self.assertFalse(score_episode(task, ep, "mock:judge")["correct"])

    def test_withheld_feedback_hides_results(self):
        seen = []
        inner = make_solver()
        llm.MockBackend.handlers["solver"] = lambda s, m: (seen.append(m[-1]["content"]), inner(s, m))[1]
        run_episode(self.build_task(), "C1", "mock:solver", "mock:gatekeeper", feedback="withheld")
        self.assertFalse(any("shows pattern" in s for s in seen))


class CalibrationTests(Base):
    def test_gates(self):
        task = calibrate_task(self.build_task(), self.cfg)
        status = {i["cause_id"]: i["status"] for i in task["instances"]}
        # The solver guesses C1 from the opening alone, so that instance is set aside.
        self.assertEqual(status, {"C1": "opening_solved", "C2": "accepted", "C3": "accepted"})
        self.assertEqual(task["status"], "accepted")
        self.assertTrue(task["calibration"]["judge_sanity"]["passed"])

    def test_unsolvable_instance_is_rejected(self):
        inner = make_solver()
        def confused(system, messages):
            r = json.loads(inner(system, messages))
            if r["action"].get("diagnosis") == "ANSWER-C3":
                r["action"]["diagnosis"] = "ANSWER-C2"
            return json.dumps(r)
        llm.MockBackend.handlers["solver"] = confused
        task = calibrate_task(self.build_task(), self.cfg)
        c3 = next(i for i in task["instances"] if i["cause_id"] == "C3")
        self.assertEqual((c3["status"], c3["reasons"]), ("rejected", ["not_solvable_from_evidence"]))

    def test_leaking_gatekeeper_is_caught(self):
        def leaky(system, messages):
            if "root cause" in messages[0]["content"]:
                truth = system.split("truth is ")[1][:2]
                return json.dumps({"kind": "result", "matched_test": None, "observation": f"ANSWER-{truth}"})
            return gatekeeper(system, messages)
        llm.MockBackend.handlers["gatekeeper"] = leaky
        task = calibrate_task(self.build_task(), self.cfg)
        self.assertTrue(all("gatekeeper_leak" in i["reasons"] for i in task["instances"]))
        self.assertEqual(task["status"], "rejected")

    def test_judge_that_cannot_separate_causes_rejects_task(self):
        llm.MockBackend.handlers["judge"] = lambda s, m: json.dumps({"asserted": "C1", "rationale": ""})
        task = calibrate_task(self.build_task(), self.cfg)
        self.assertEqual((task["status"], task["reject_reason"]), ("rejected", "judge_cannot_score"))


if __name__ == "__main__":
    unittest.main()
