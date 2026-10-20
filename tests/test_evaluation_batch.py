from scripts.evaluate_stage_batch import summarize


def test_evaluation_report_separates_visits_success_and_table_safety():
    results=[dict(success=False,trace=[dict(stage=3,measurements={'slide_distance':.12},termination=['unsafe_table_contact'])]),
             dict(success=True,trace=[dict(stage=2,measurements={'slide_distance':.004})])]
    r=summarize(results)
    assert r['successes']==1 and r['reached_place']==1
    assert r['unsafe_table_episodes']==1
    assert abs(r['mean_table_slide_m']-.062)<1e-6
