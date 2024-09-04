from vllm.mlu_hijack.mlu_hijack_utils import MluHijackObject

from vllm.logger import init_logger
logger = init_logger(__name__)

import os
def _check_env(env, default=False):
    if env in os.environ:
        return os.environ[env].lower() in ["true", "1"]
    return default
VLLM_SCHEDULER_PROFILE = _check_env("VLLM_SCHEDULER_PROFILE", default=False)


class VllmMluHijack:
    def __init__(self):
        pass

    def apply_optimizations(self):
        logger.info("Apply MLU optimization!")
        if VLLM_SCHEDULER_PROFILE:
            import vllm.mlu_hijack.core.scheduler
            import vllm.mlu_hijack.engine.async_llm_engine
            import vllm.mlu_hijack.entrypoints.openai.serving_engine
        MluHijackObject.apply_hijack()

    def undo_optimization(self):
        logger.info("Undo MLU optimization!")
        MluHijackObject.undo_hijack()

vllm_mlu_hijack = VllmMluHijack()
vllm_mlu_hijack.apply_optimizations()
