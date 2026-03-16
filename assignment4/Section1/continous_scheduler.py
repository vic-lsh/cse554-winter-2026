from continous_engine import Engine, Request
import torch

class InputRequest:
    def __init__(self, input_str: str, output_len: int):
        self.input_str = input_str
        self.output_len = output_len
        
class CBScheduler:
    def __init__(self, engine: Engine, req_batch_size: int):
        self.engine = engine
        self.req_batch_size = req_batch_size
        self.pending_input_req: list[InputRequest] = []
        self.decode_req: list[Request] = []
        self.scheduled_prefill_req: list[Request] = []
        self.completed: list[Request] = []
        self.unique_req_id: int = 0
    
    def add_req(self, input_req: InputRequest):
        self.pending_input_req.append(input_req)
        
    def finished(self) -> bool:
        return not self.pending_input_req and not self.decode_req and not self.scheduled_prefill_req

    def get_req_batch_size(self) -> int:
        return len(self.decode_req) + len(self.scheduled_prefill_req)

    def run(self):
        # Schedule new prefill requests until batch is full or no pending inputs
        #########
        # FIXME #
        #########

        # Build the list of requests to send to the engine
        if self.finished():
            print("No pending requests to schedule or decode.")
            return

        request_list_total = []
        decode_num = 0
        while self.get_req_batch_size() < self.req_batch_size and self.pending_input_req:
            input_req = self.pending_input_req.pop(0)

            prompt_ids = self.engine.tokenizer(input_req.input_str, add_special_tokens=False).input_ids
            prompt_id_tensor = torch.tensor(prompt_ids, dtype=torch.long)

            new_req = Request(
                req_id=self.unique_req_id,
                prompt_ids=prompt_id_tensor,
                target_len=input_req.output_len
            )
            self.unique_req_id += 1
            self.scheduled_prefill_req.append(new_req)
        request_list_total = self.decode_req + self.scheduled_prefill_req
        decode_num = len(self.decode_req)


        #########
        # FIXME #
        #########
        new_tokens = self.engine.run(request_list_total, decode_num)

        # Append newly generated tokens to each request's output buffer
        # print(new_tokens.shape)
        for i, req in enumerate(request_list_total):
            req.output_token_ids = torch.cat(
                [req.output_token_ids, new_tokens[i:i+1].to(req.output_token_ids.device)],
                dim=0
            )


        #########
        # FIXME #
        #########

        # Check which decode requests have finished and remove from the queue
        ongoing_decode: list[Request] = []
        #########
        # FIXME #
        #########
        for req in request_list_total:
            if req.current_length >= req.output_length + len(req.prompt_token_ids):
                self.completed.append(req)
                cache = self.engine.kv_cache_map.pop(req.request_id, None)
                if cache is not None:
                    cache.release()
            else:
                ongoing_decode.append(req)

        # Move scheduled prefill requests into decode queue
        #########
        # FIXME #
        #########
        self.decode_req = ongoing_decode

        self.scheduled_prefill_req = []
    
    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")

        
class NaiveScheduler:
    def __init__(self, engine: Engine, req_batch_size: int):
        self.engine = engine
        self.req_batch_size = req_batch_size
        self.pending_input_req: list[InputRequest] = []
        self.decode_req: list[Request] = []
        self.scheduled_prefill_req: list[Request] = []
        self.completed: list[Request] = []
        self.unique_req_id: int = 0
    
    def add_req(self, input_req: InputRequest):
        self.pending_input_req.append(input_req)
        
    def finished(self) -> bool:
        return not self.pending_input_req and not self.decode_req and not self.scheduled_prefill_req

    def get_req_batch_size(self) -> int:
        return len(self.decode_req) + len(self.scheduled_prefill_req)

    def run(self):
        # Schedule new prefill requests until batch is full or no pending inputs
        #########
        # FIXME #
        #########

        # Build the list of requests to send to the engine
        if self.finished():
            print("No pending requests to schedule or decode.")
            return

        request_list_total = []
        decode_num = 0
        if not self.decode_req and not self.scheduled_prefill_req:
            while self.get_req_batch_size() < self.req_batch_size and self.pending_input_req:
                input_req = self.pending_input_req.pop(0)

                prompt_ids = self.engine.tokenizer(input_req.input_str, add_special_tokens=False).input_ids
                prompt_id_tensor = torch.tensor(prompt_ids, dtype=torch.long)

                new_req = Request(
                    req_id=self.unique_req_id,
                    prompt_ids=prompt_id_tensor,
                    target_len=input_req.output_len
                )
                self.unique_req_id += 1
                self.scheduled_prefill_req.append(new_req)
        request_list_total = self.decode_req + self.scheduled_prefill_req
        decode_num = len(self.decode_req)


        #########
        # FIXME #
        #########
        new_tokens = self.engine.run(request_list_total, decode_num)

        # Append newly generated tokens to each request's output buffer
        # print(new_tokens.shape)
        for i, req in enumerate(request_list_total):
            req.output_token_ids = torch.cat(
                [req.output_token_ids, new_tokens[i:i+1].to(req.output_token_ids.device)],
                dim=0
            )


        #########
        # FIXME #
        #########

        # Check which decode requests have finished and remove from the queue
        ongoing_decode: list[Request] = []
        #########
        # FIXME #
        #########
        for req in request_list_total:
            if req.current_length >= req.output_length + len(req.prompt_token_ids):
                self.completed.append(req)
                cache = self.engine.kv_cache_map.pop(req.request_id, None)
                if cache is not None:
                    cache.release()
            else:
                ongoing_decode.append(req)

        # Move scheduled prefill requests into decode queue
        #########
        # FIXME #
        #########
        self.decode_req = ongoing_decode
        self.scheduled_prefill_req = []
    
    def print_completed(self):
        for i, req in enumerate(self.completed):
            text = self.engine.tokenizer.decode(
                req.output_token_ids, skip_special_tokens=True
            )
            print(f"Id = {i}: {text}")
