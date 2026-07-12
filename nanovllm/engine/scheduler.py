from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.enable_trunked_prefill=config.enable_chunked_prefill
        self.enable_mixed_prefill_decode=config.enable_mixed_prefill_decode
        self.prefill_chunk_size=config.prefill_chunk_size
    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> ScheduleOut:
        scheduled_seqs = []
        token_buget=self.max_num_batched_tokens
        seq_buget=self.max_num_seqs
        num_batched_tokens = 0
        num_prefill_tokens=0
        num_decode_tokens=0
        has_prefill=False
        has_decode=False
        if self.enable_trunked_prefill:
            num_running=len(self.running)
            for _ in range(num_running):
                if not self.running:
                    break
                if token_buget<=0 or len(scheduled_seqs)>=seq_buget:
                    break
                seq=self.running.popleft()
                assert seq.num_cached_tokens==len(seq)-1,(
                    seq.seq_id,
                    seq.num_cached_tokens,
                    len(seq)
                )
                while not self.block_manager.can_append(seq):
                    if self.running:
                        victim=self.running.pop()
                        self.preempt(victim)
                    else:
                        self.preempt(seq)
                        seq=None
                        break
                if seq is None:
                    continue
                seq.is_prefill=False
                seq.num_scheduled_tokens=1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                self.running.append(seq)
                token_buget-=1
                num_decode_tokens+=1
                has_decode=True
            while self.waiting and token_buget>0 and len(scheduled_seqs)<seq_buget:
                seq=self.waiting[0]
                seq.is_prefill=True
                if not seq.block_table:
                    num_cached_blocks=self.block_manager.get_cached_prefix(seq)
                    self.block_manager.attach_cached_prefix(seq,num_cached_blocks)
                target_len=len(seq)
                remaining_len=target_len-num_cached_blocks
                assert remaining_len>0
                chunk=min(remaining_len,token_buget,self.prefll_chunk_size)
                end=chunk+num_cached_blocks
                if not self.block_manager.can_allocat_tokens(seq,end):
                    break
                self.block_manager.ensure_allocate(seq,end)
                seq.num_scheduled_tokens=chunk
                scheduled_seqs.append(seq)
                token_buget-=chunk
                num_prefill_tokens+=chunk
                has_prefill=True
                if end==target_len:
                    self.waiting.popleft()
                    seq.status=SequenceStatus.RUNNING
                    self.running.append(seq)
                else:
                    break
            assert scheduled_seqs
            return ScheduleOut(
                has_prefill,
                has_decode,
                num_prefill_tokens,
                num_decode_tokens,
                scheduled_seqs,
                num_batched_tokens
            )

        else:
            # prefill
            while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.waiting[0]
                remaining = self.max_num_batched_tokens - num_batched_tokens
                if remaining == 0:
                    break
                if not seq.block_table:
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        break
                    num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                else:
                    num_tokens = seq.num_tokens - seq.num_cached_tokens
                if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                    break
                if not seq.block_table:
                    self.block_manager.allocate(seq, num_cached_blocks)
                seq.num_scheduled_tokens = min(num_tokens, remaining)
                num_batched_tokens += seq.num_scheduled_tokens
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                    seq.status = SequenceStatus.RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)
                scheduled_seqs.append(seq)

            if scheduled_seqs:
                return scheduled_seqs, True

            # decode
            while self.running and len(scheduled_seqs) < self.max_num_seqs:
                seq = self.running.popleft()
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
            assert scheduled_seqs
            self.running.extendleft(reversed(scheduled_seqs))
            return ScheduleOut(
                has_prefill,
                has_decode,
                num_prefill_tokens,
                num_decode_tokens,
                scheduled_seqs,
                num_batched_tokens
            )

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)


class ScheduleOut:
    has_prefill: bool
    has_decode: bool
    num_prefill_tokens: int
    num_decode_tokens: int
    seqs: list[Sequence]
    num_batched_tokens: int