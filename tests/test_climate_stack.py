from __future__ import annotations

import unittest
from threading import Lock
from time import sleep
from datetime import date
from unittest.mock import patch

import pandas as pd

from mystripes.cds import ERA5_LAND_MONTHLY_DATASET, ERA5_MONTHLY_DATASET
from mystripes.climate_stack import (
    CALIBRATION_BASELINE_END,
    CALIBRATION_BASELINE_START,
    ClimateBatchPlan,
    ClimateDownloadEstimate,
    LocationClimateRequest,
    build_climate_batch_plan,
    build_climate_source_slices,
    climate_stack_note,
    estimate_climate_downloads,
    fetch_saved_climate_series_batch,
    fetch_saved_climate_series,
    get_climate_dataset_window,
    preflight_download_message,
)
from mystripes.models import CDSConfig, DatasetWindow


def _monthly_frame(start: str, periods: int, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq="MS", tz="UTC"),
            "temperature_c": values,
            "sample_days": [31] * periods,
        }
    )


class ClimateStackTests(unittest.TestCase):
    def test_get_climate_dataset_window_extends_to_pre_era5_start(self) -> None:
        with patch(
            "mystripes.climate_stack.get_dataset_window",
            return_value=DatasetWindow(min_start=date(1950, 1, 1), max_end=date(2026, 3, 31)),
        ):
            window = get_climate_dataset_window()

        self.assertEqual(window.min_start, date(1836, 1, 1))
        self.assertEqual(window.max_end, date(2026, 3, 31))

    def test_build_climate_source_slices_uses_full_stack(self) -> None:
        source_slices = build_climate_source_slices(date(1938, 1, 1), date(1951, 12, 31))

        self.assertEqual(
            [(source.source_id, source.start_date, source.end_date) for source in source_slices],
            [
                ("20crv3", date(1938, 1, 1), date(1939, 12, 31)),
                ("era5_bridge", date(1940, 1, 1), date(1949, 12, 31)),
                ("era5_land", date(1950, 1, 1), date(1951, 12, 31)),
            ],
        )

    def test_climate_stack_note_mentions_historical_sources(self) -> None:
        self.assertIsNone(climate_stack_note(date(1955, 1, 1), date(2000, 12, 31)))
        note = climate_stack_note(date(1938, 1, 1), date(1945, 12, 31))
        self.assertIn("ERA5 monthly single levels", note)
        self.assertIn("20CRv3", note)
        self.assertIn("1961-1990", note)

    def test_preflight_download_message_warns_for_large_historical_work(self) -> None:
        level, message = preflight_download_message(
            ClimateDownloadEstimate(
                uses_historical_fallback=True,
                uncached_era5_bridge_fetches=3,
                uncached_twcr_fetches=12,
                uncached_twcr_years=tuple(range(1836, 1848)),
            )
        )
        self.assertEqual(level, "warning")
        self.assertIn("12 uncached 20CRv3 bbox downloads", message)
        self.assertIn("Later runs reuse the local cache", message)

    def test_estimate_climate_downloads_counts_bridge_fetches_and_twcr_years(self) -> None:
        request = LocationClimateRequest(
            location_key="48.2,16.4",
            display_name="Vienna",
            latitude=48.2,
            longitude=16.4,
            boundary_geojson=None,
            boundary_bbox=None,
            start_date=date(1938, 1, 1),
            end_date=date(1945, 12, 31),
        )

        with patch(
            "mystripes.climate_stack.build_climate_batch_plan",
            return_value=type(
                "Plan",
                (),
                {
                    "location_requests": (request,),
                    "shared_tasks": (
                        type("Task", (), {"source_id": "era5_bridge", "year": None})(),
                        type("Task", (), {"source_id": "era5_bridge", "year": None})(),
                        type(
                            "Task",
                            (),
                            {"source_id": "20crv3", "year": None, "start_date": date(1938, 1, 1), "end_date": date(1939, 12, 31)},
                        )(),
                        type(
                            "Task",
                            (),
                            {"source_id": "20crv3", "year": None, "start_date": date(1961, 1, 1), "end_date": date(1962, 12, 31)},
                        )(),
                    ),
                },
            )(),
        ):
            estimate = estimate_climate_downloads([request], spatial_mode="single_cell")

        self.assertTrue(estimate.uses_historical_fallback)
        self.assertEqual(estimate.uncached_era5_bridge_fetches, 2)
        self.assertEqual(estimate.uncached_twcr_fetches, 2)
        self.assertEqual(estimate.uncached_twcr_years, (1938, 1939, 1961, 1962))

    def test_build_climate_batch_plan_deduplicates_shared_era5_tasks(self) -> None:
        requests = [
            LocationClimateRequest(
                location_key="a",
                display_name="A",
                latitude=48.2082,
                longitude=16.3738,
                boundary_geojson=None,
                boundary_bbox=None,
                start_date=date(1944, 1, 1),
                end_date=date(1944, 12, 31),
            ),
            LocationClimateRequest(
                location_key="b",
                display_name="B",
                latitude=48.2401,
                longitude=16.3601,
                boundary_geojson=None,
                boundary_bbox=None,
                start_date=date(1944, 1, 1),
                end_date=date(1944, 12, 31),
            ),
        ]

        with patch("mystripes.climate_stack._load_cached_temperature_series", return_value=None), patch(
            "pathlib.Path.exists",
            return_value=False,
        ), patch(
            "mystripes.climate_stack._needs_cds_window_fetch",
            return_value=False,
        ):
            batch_plan = build_climate_batch_plan(requests, spatial_mode="single_cell")

        era5_bridge_tasks = [task for task in batch_plan.shared_tasks if task.source_id == "era5_bridge"]
        visible_tasks = [
            task
            for task in era5_bridge_tasks
            if task.start_date == date(1944, 1, 1) and task.end_date == date(1944, 12, 31)
        ]
        self.assertEqual(len(visible_tasks), 1)

    def test_build_climate_batch_plan_uses_twcr_subset_tasks_for_small_requests(self) -> None:
        requests = [
            LocationClimateRequest(
                location_key="a",
                display_name="A",
                latitude=48.2082,
                longitude=16.3738,
                boundary_geojson=None,
                boundary_bbox=None,
                start_date=date(1938, 1, 1),
                end_date=date(1938, 12, 31),
            ),
        ]

        with patch("mystripes.climate_stack._load_cached_temperature_series", return_value=None), patch(
            "pathlib.Path.exists",
            return_value=False,
        ), patch(
            "mystripes.climate_stack._needs_cds_window_fetch",
            return_value=False,
        ):
            batch_plan = build_climate_batch_plan(requests, spatial_mode="single_cell")

        visible_twcr_tasks = [
            task
            for task in batch_plan.shared_tasks
            if task.source_id == "20crv3"
            and task.task_kind == "twcr_grid"
            and task.start_date == date(1938, 1, 1)
            and task.end_date == date(1990, 12, 31)
        ]
        self.assertEqual(len(visible_twcr_tasks), 1)
        self.assertIn(visible_twcr_tasks[0].work_id, batch_plan.location_dependencies["a"])
        self.assertEqual(batch_plan.location_twcr_tasks["a"][0].work_id, visible_twcr_tasks[0].work_id)

    def test_build_climate_batch_plan_merges_nearby_twcr_requests_into_shared_bbox(self) -> None:
        requests = [
            LocationClimateRequest(
                location_key="vienna",
                display_name="Vienna",
                latitude=48.2082,
                longitude=16.3738,
                boundary_geojson=None,
                boundary_bbox=None,
                start_date=date(1938, 1, 1),
                end_date=date(1938, 12, 31),
            ),
            LocationClimateRequest(
                location_key="bratislava",
                display_name="Bratislava",
                latitude=48.1486,
                longitude=17.1077,
                boundary_geojson=None,
                boundary_bbox=None,
                start_date=date(1938, 1, 1),
                end_date=date(1938, 12, 31),
            ),
        ]

        with patch("mystripes.climate_stack._load_cached_temperature_series", return_value=None), patch(
            "pathlib.Path.exists",
            return_value=False,
        ), patch(
            "mystripes.climate_stack._needs_cds_window_fetch",
            return_value=False,
        ):
            batch_plan = build_climate_batch_plan(requests, spatial_mode="single_cell")

        twcr_tasks = [task for task in batch_plan.shared_tasks if task.source_id == "20crv3"]
        self.assertEqual(len(twcr_tasks), 1)
        self.assertEqual(twcr_tasks[0].area, (48.0, 16.0, 48.0, 17.0))
        self.assertIn(twcr_tasks[0].work_id, batch_plan.location_dependencies["vienna"])
        self.assertIn(twcr_tasks[0].work_id, batch_plan.location_dependencies["bratislava"])

    def test_fetch_saved_climate_series_batch_reports_shared_and_location_progress(self) -> None:
        request = LocationClimateRequest(
            location_key="a",
            display_name="A",
            latitude=48.2082,
            longitude=16.3738,
            boundary_geojson=None,
            boundary_bbox=None,
            start_date=date(1955, 1, 1),
            end_date=date(1955, 12, 31),
        )
        events: list[dict[str, object]] = []
        frame = _monthly_frame("1955-01-01", 1, [1.0])

        with patch(
            "mystripes.climate_stack.build_climate_batch_plan",
            return_value=ClimateBatchPlan(location_requests=(request,), shared_tasks=(), location_dependencies={}),
        ), patch(
            "mystripes.climate_stack.fetch_saved_climate_series",
            return_value=frame,
        ):
            result = fetch_saved_climate_series_batch(
                config=CDSConfig(url="https://example.invalid/api", key="secret"),
                location_requests=[request],
                progress_callback=events.append,
            )

        self.assertIn("a", result)
        self.assertEqual(events[0]["stage"], "batch_plan")
        self.assertEqual(events[1]["stage"], "location_started")
        self.assertEqual(events[2]["stage"], "location_finished")

    def test_fetch_saved_climate_series_batch_starts_ready_locations_before_all_shared_tasks_finish(self) -> None:
        request_a = LocationClimateRequest(
            location_key="a",
            display_name="A",
            latitude=48.2082,
            longitude=16.3738,
            boundary_geojson=None,
            boundary_bbox=None,
            start_date=date(1955, 1, 1),
            end_date=date(1955, 12, 31),
        )
        request_b = LocationClimateRequest(
            location_key="b",
            display_name="B",
            latitude=52.52,
            longitude=13.405,
            boundary_geojson=None,
            boundary_bbox=None,
            start_date=date(1955, 1, 1),
            end_date=date(1955, 12, 31),
        )
        task_a = type(
            "Task",
            (),
            {"work_id": "shared-a", "source_id": "era5_land", "label": "ERA5", "task_kind": "cds_grid", "year": None, "area": None, "dataset": None, "start_date": None, "end_date": None},
        )()
        task_b = type(
            "Task",
            (),
            {"work_id": "shared-b", "source_id": "era5_land", "label": "ERA5", "task_kind": "cds_grid", "year": None, "area": None, "dataset": None, "start_date": None, "end_date": None},
        )()
        execution_log: list[str] = []
        log_lock = Lock()
        frame = _monthly_frame("1955-01-01", 1, [1.0])

        def fake_run_shared_task(config, task, progress_callback):
            with log_lock:
                execution_log.append(f"shared_start:{task.work_id}")
            sleep(0.02 if task.work_id == "shared-a" else 0.25)
            with log_lock:
                execution_log.append(f"shared_finish:{task.work_id}")

        def fake_run_location_task(config, request, spatial_mode, radius_km, twcr_request_tasks, progress_callback):
            with log_lock:
                execution_log.append(f"location_start:{request.location_key}")
            return frame

        with patch(
            "mystripes.climate_stack.build_climate_batch_plan",
            return_value=ClimateBatchPlan(
                location_requests=(request_a, request_b),
                shared_tasks=(task_a, task_b),
                location_dependencies={"a": ("shared-a",), "b": ("shared-b",)},
            ),
        ), patch(
            "mystripes.climate_stack._run_shared_task",
            side_effect=fake_run_shared_task,
        ), patch(
            "mystripes.climate_stack._run_location_task",
            side_effect=fake_run_location_task,
        ):
            result = fetch_saved_climate_series_batch(
                config=CDSConfig(url="https://example.invalid/api", key="secret"),
                location_requests=[request_a, request_b],
                max_workers=3,
            )

        self.assertEqual(set(result), {"a", "b"})
        self.assertLess(
            execution_log.index("location_start:a"),
            execution_log.index("shared_finish:shared-b"),
        )

    def test_fetch_saved_climate_series_batch_falls_back_to_pending_locations(self) -> None:
        request = LocationClimateRequest(
            location_key="a",
            display_name="A",
            latitude=48.2082,
            longitude=16.3738,
            boundary_geojson=None,
            boundary_bbox=None,
            start_date=date(1955, 1, 1),
            end_date=date(1955, 12, 31),
        )
        frame = _monthly_frame("1955-01-01", 1, [1.0])

        with patch(
            "mystripes.climate_stack.build_climate_batch_plan",
            return_value=ClimateBatchPlan(
                location_requests=(request,),
                shared_tasks=(),
                location_dependencies={"a": ("missing-shared-task",)},
            ),
        ), patch(
            "mystripes.climate_stack._run_location_task",
            return_value=frame,
        ) as run_location_task_mock:
            result = fetch_saved_climate_series_batch(
                config=CDSConfig(url="https://example.invalid/api", key="secret"),
                location_requests=[request],
            )

        self.assertEqual(set(result), {"a"})
        self.assertEqual(run_location_task_mock.call_count, 1)

    def test_fetch_saved_climate_series_keeps_modern_era5_land_data_unchanged(self) -> None:
        land_frame = _monthly_frame("1955-01-01", 2, [1.0, 2.0])

        with patch(
            "mystripes.climate_stack.fetch_saved_temperature_series",
            return_value=land_frame,
        ) as fetch_saved_temperature_series_mock:
            combined = fetch_saved_climate_series(
                config=CDSConfig(url="https://example.invalid/api", key="secret"),
                latitude=48.2,
                longitude=16.4,
                start_date=date(1955, 1, 1),
                end_date=date(1955, 2, 28),
            )

        pd.testing.assert_frame_equal(combined, land_frame)
        dataset = fetch_saved_temperature_series_mock.call_args.kwargs["dataset"]
        self.assertEqual(dataset.name, ERA5_LAND_MONTHLY_DATASET.name)

    def test_fetch_saved_climate_series_calibrates_era5_bridge_slice(self) -> None:
        visible_frame = _monthly_frame("1944-01-01", 2, [5.0, 6.0])
        anchor_frame = _monthly_frame("1961-01-01", 2, [12.0, 13.0])
        source_calibration_frame = _monthly_frame("1961-01-01", 2, [10.0, 11.0])

        def fake_fetch_saved_temperature_series(**kwargs):
            dataset = kwargs["dataset"]
            if dataset.name == ERA5_MONTHLY_DATASET.name:
                return visible_frame
            raise AssertionError(f"Unexpected saved fetch dataset: {dataset.name}")

        def fake_fetch_point_temperature_series(**kwargs):
            dataset = kwargs["dataset"]
            if dataset.name == ERA5_LAND_MONTHLY_DATASET.name:
                return anchor_frame
            if dataset.name == ERA5_MONTHLY_DATASET.name:
                return source_calibration_frame
            raise AssertionError(f"Unexpected point fetch dataset: {dataset.name}")

        with patch(
            "mystripes.climate_stack.fetch_saved_temperature_series",
            side_effect=fake_fetch_saved_temperature_series,
        ), patch(
            "mystripes.climate_stack.fetch_point_temperature_series",
            side_effect=fake_fetch_point_temperature_series,
        ):
            combined = fetch_saved_climate_series(
                config=CDSConfig(url="https://example.invalid/api", key="secret"),
                latitude=48.2,
                longitude=16.4,
                start_date=date(1944, 1, 1),
                end_date=date(1944, 2, 29),
            )

        self.assertEqual(combined["temperature_c"].round(2).tolist(), [7.0, 8.0])

    def test_fetch_saved_climate_series_calibrates_twcr_slice(self) -> None:
        combined_twcr_frame = pd.concat(
            [
                _monthly_frame("1938-01-01", 2, [4.0, 5.0]),
                _monthly_frame("1961-01-01", 2, [9.0, 10.0]),
            ],
            ignore_index=True,
        )
        anchor_frame = _monthly_frame("1961-01-01", 2, [12.0, 13.0])

        with patch(
            "mystripes.climate_stack.fetch_saved_twcr_temperature_series",
            return_value=combined_twcr_frame,
        ), patch(
            "mystripes.climate_stack.fetch_point_temperature_series",
            return_value=anchor_frame,
        ) as fetch_point_temperature_series_mock:
            combined = fetch_saved_climate_series(
                config=CDSConfig(url="https://example.invalid/api", key="secret"),
                latitude=48.2,
                longitude=16.4,
                start_date=date(1938, 1, 1),
                end_date=date(1938, 2, 28),
            )

        self.assertEqual(combined["temperature_c"].round(2).tolist(), [7.0, 8.0])
        self.assertEqual(fetch_point_temperature_series_mock.call_count, 1)

    def test_fetch_saved_climate_series_falls_back_to_scale_one_when_variance_is_missing(self) -> None:
        combined_twcr_frame = pd.concat(
            [
                _monthly_frame("1938-01-01", 1, [4.0]),
                _monthly_frame("1961-01-01", 1, [9.0]),
            ],
            ignore_index=True,
        )
        anchor_frame = _monthly_frame("1961-01-01", 1, [12.0])

        with patch(
            "mystripes.climate_stack.fetch_saved_twcr_temperature_series",
            return_value=combined_twcr_frame,
        ), patch(
            "mystripes.climate_stack.fetch_point_temperature_series",
            return_value=anchor_frame,
        ):
            combined = fetch_saved_climate_series(
                config=CDSConfig(url="https://example.invalid/api", key="secret"),
                latitude=48.2,
                longitude=16.4,
                start_date=date(1938, 1, 1),
                end_date=date(1938, 1, 31),
            )

        self.assertEqual(combined["temperature_c"].round(2).tolist(), [7.0])

    def test_location_request_dataclass_stays_constructible_for_estimate_helpers(self) -> None:
        request = LocationClimateRequest(
            location_key="48.2,16.4",
            display_name="Vienna",
            latitude=48.2,
            longitude=16.4,
            boundary_geojson=None,
            boundary_bbox=None,
            start_date=CALIBRATION_BASELINE_START,
            end_date=CALIBRATION_BASELINE_END,
        )
        self.assertEqual(request.display_name, "Vienna")


if __name__ == "__main__":
    unittest.main()
