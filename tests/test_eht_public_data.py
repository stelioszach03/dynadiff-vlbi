from __future__ import annotations

from pathlib import Path

import numpy as np

from dynadiff_vlbi.data.eht_public_data import (
    get_public_eht_release_spec,
    load_public_eht_csv,
    prepare_public_eht_validation_dataset,
    prepare_public_m87_validation_dataset,
)


def _write_public_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "#SRC:M87,DATE(MJD):57848,FREQ:229.0707GHz",
                "#time(UTC),T1,T2,U(lambda),V(lambda),Iamp(Jy),Iphase(d),Isigma(Jy)",
                "0.10,AP,AA,10.0,5.0,1.00,45.0,0.10",
                "0.10,AA,AZ,-7.0,4.0,0.80,-15.0,0.20",
                "0.10,AP,AZ,6.0,-3.0,0.65,10.0,0.15",
                "0.20,AA,AP,11.0,6.0,1.10,40.0,0.10",
                "0.20,AZ,AA,7.5,-4.5,0.75,15.0,0.20",
                "0.20,AZ,AP,-6.5,3.5,0.60,-12.0,0.15",
                "0.30,AP,AA,12.0,6.5,1.05,48.0,0.10",
                "0.30,AA,AZ,-8.0,4.5,0.82,-18.0,0.20",
                "0.30,AP,AZ,7.0,-3.5,0.62,6.0,0.15",
                "0.40,AA,AP,12.5,7.0,1.02,43.0,0.10",
                "0.40,AZ,AA,8.0,-4.0,0.77,12.0,0.20",
                "0.40,AZ,AP,-7.0,3.0,0.58,-8.0,0.15",
            ]
        ),
        encoding="utf-8",
    )


def test_load_public_eht_csv_canonicalizes_station_order(tmp_path: Path) -> None:
    path = tmp_path / "SR1_M87_2017_095_hi_hops_netcal_StokesI.csv"
    _write_public_csv(path)
    metadata, records = load_public_eht_csv(path)
    assert metadata["SRC"] == "M87"
    assert records[0].station_a == "AA"
    assert records[0].station_b == "AP"
    assert np.isclose(records[0].u_lambda, -10.0)
    assert np.isclose(records[0].v_lambda, -5.0)


def test_prepare_public_m87_validation_dataset_writes_expected_arrays(tmp_path: Path) -> None:
    source_root = tmp_path / "public_eht"
    csv_path = source_root / "csv" / "SR1_M87_2017_095_hi_hops_netcal_StokesI.csv"
    _write_public_csv(csv_path)

    manifest = prepare_public_m87_validation_dataset(
        source_root=source_root,
        output_dir=tmp_path / "prepared",
        image_size=8,
        sequence_length=4,
    )

    assert manifest["sample_count"] == 1
    with np.load(tmp_path / "prepared" / "test.npz") as payload:
        assert payload["vis_real"].shape == (1, 4, 8, 8)
        assert payload["mask"].shape == (1, 4, 8, 8)
        assert payload["dirty"].shape == (1, 4, 8, 8)
        assert payload["baseline_pairs"].shape == (3, 2)
        assert payload["frame_uv_indices"].shape == (1, 4, 3, 2)
        assert payload["sample_id"][0] == "M87_2017_095_hi"


def test_prepare_public_release_dataset_supports_2018_m87_and_sigma_arrays(tmp_path: Path) -> None:
    source_root = tmp_path / "public_eht_2018"
    csv_path = source_root / "csv" / "L2V1_M87_2018_111_b1_hops_netcal_10s_StokesI.csv"
    _write_public_csv(csv_path)

    manifest = prepare_public_eht_validation_dataset(
        source_root=source_root,
        output_dir=tmp_path / "prepared_2018",
        release_code="2024-D01-01",
        image_size=8,
        sequence_length=4,
    )

    spec = get_public_eht_release_spec("2024-D01-01")
    assert manifest["release_code"] == spec.release_code
    assert manifest["target"] == "M87"
    with np.load(tmp_path / "prepared_2018" / "test.npz") as payload:
        assert payload["sample_id"][0] == "M87_2018_111_b1"
        assert payload["release_code"][0] == "2024-D01-01"
        assert payload["vis_sigma"].shape == (1, 4, 8, 8)
        assert payload["vis_weight"].shape == (1, 4, 8, 8)
        assert np.count_nonzero(payload["vis_sigma"]) > 0
        assert np.count_nonzero(payload["vis_weight"]) > 0
