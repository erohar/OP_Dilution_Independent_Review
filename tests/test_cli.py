import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import check_model
import run_dilution

def test_missing_model_logs_error_and_exits_1(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        rc = run_dilution.cli([str(tmp_path / "missing.csv"), "--out", str(tmp_path),
                               "--cutoff", "0.5", "--smu-x", "10", "--step-x", "1",
                               "--step-y", "1"])
    assert rc == 1
    assert "Block model not found" in caplog.text

def test_missing_column_logs_error_and_cleans_work_files(tmp_path, caplog):
    model = tmp_path / "m.csv"
    model.write_text("XC,YC,ZC,AU,RESCAT,DENSITY\n2.5,2.5,2.5,1.0,1,2.7\n")
    out = tmp_path / "out"
    with caplog.at_level(logging.ERROR):
        rc = run_dilution.cli([str(model), "--out", str(out), "--cutoff", "0.5",
                               "--smu-x", "5", "--step-x", "5", "--step-y", "5"])
    assert rc == 1
    assert "OXIDE" in caplog.text
    assert not any(n.startswith(".dilution_tmp") for n in os.listdir(out))

def test_bad_settings_json_logs_error(tmp_path, caplog):
    bad = tmp_path / "s.json"
    bad.write_text("{not json")
    with caplog.at_level(logging.ERROR):
        assert check_model.main(["--settings", str(bad)]) == 1
    assert "not valid JSON" in caplog.text

def test_missing_settings_file_logs_error(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        assert check_model.main(["--settings", str(tmp_path / "none.json")]) == 1
    assert "Settings file not found" in caplog.text

def test_region_and_edge_dilution_options(tmp_path):
    regions = tmp_path / "r.csv"
    regions.write_text("REGION,PRIORITY,X1,Y1,X2,Y2,X3,Y3,X4,Y4\n"
                       "A,1,0,10,10,10,10,0,0,0\n")
    a = run_dilution.build_parser().parse_args(
        ["m.csv", "--out", str(tmp_path), "--cutoff", "0.5", "--smu-x", "10",
         "--step-x", "1", "--step-y", "1", "--threads", "1", "--regions", str(regions),
         "--elos", "2", "--ore-rule", "value", "--outside", "waste",
         "--audit-fields"])
    p = run_dilution.params_from_args(a)
    assert [r.name for r in p.regions] == ["A"]
    assert (p.elos, p.ore_rule, p.outside, p.audit_fields) == (2.0, "value", "waste", True)
    assert p.step_y == 1.0
    p.validate()
