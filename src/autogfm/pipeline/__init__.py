"""End-to-end project pipelines."""

from autogfm.pipeline.stage3 import (
    Stage3PipelineResult,
    Stage3PipelineSpec,
    run_stage3_pipeline,
)
from autogfm.pipeline.stage4 import (
    Stage4PipelineResult,
    Stage4PipelineSpec,
    run_stage4_pipeline,
)
from autogfm.pipeline.stage10a import Stage10AResult, run_stage10a_real_market
from autogfm.pipeline.stage11 import Stage11Result, Stage11Spec, run_stage11_pipeline
from autogfm.pipeline.stage11b import (
    Stage11BResult,
    Stage11BSpec,
    run_stage11b_pipeline,
)
from autogfm.pipeline.stage12a import (
    Stage12AResult,
    Stage12ASpec,
    run_stage12a_pipeline,
)
from autogfm.pipeline.stage5 import (
    Stage5PipelineResult,
    Stage5PipelineSpec,
    run_stage5_pipeline,
)

__all__ = [
    "Stage3PipelineResult",
    "Stage3PipelineSpec",
    "Stage4PipelineResult",
    "Stage4PipelineSpec",
    "Stage10AResult",
    "Stage11Result",
    "Stage11Spec",
    "Stage11BResult",
    "Stage11BSpec",
    "Stage12AResult",
    "Stage12ASpec",
    "Stage5PipelineResult",
    "Stage5PipelineSpec",
    "run_stage3_pipeline",
    "run_stage4_pipeline",
    "run_stage10a_real_market",
    "run_stage11_pipeline",
    "run_stage11b_pipeline",
    "run_stage12a_pipeline",
    "run_stage5_pipeline",
]
