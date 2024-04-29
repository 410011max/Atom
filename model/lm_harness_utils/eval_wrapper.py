# Import necessary modules
import time
import torch
import torch.nn as nn

from collections import defaultdict
import fnmatch



from transformers import AutoTokenizer, AutoModelForCausalLM

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

from lm_eval.utils import make_table

@register_model("llama")
class LLaMAEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = AutoModelForCausalLM

    def __init__(self, model, tokenizer, max_length=2048, batch_size=1, device="cuda"):
        super().__init__()
        LM.__init__(self)
        self._model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size)
        self._max_length = max_length
        self._device = torch.device(device)
        self._model = self._model.to(self._device)

    @property
    def batch_size(self):
        return self._batch_size



def eval_llama_zero_shot(model, tokenizer, batch_size=1, max_length=2048, task_list=["wikitext"], num_fewshot=0):
    import lm_eval
    import os
    # Workaround for the following error
    # huggingface/tokenizers: The current process just got forked, 
    # after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    lm_obj = LLaMAEvalWrapper(model=model, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager is the it is set to None here.
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=task_list,
        num_fewshot=num_fewshot,
        device="cuda",
        batch_size=batch_size,
        task_manager=task_manager,
        log_samples=False
    ) 
    print(make_table(results))
    
    return results