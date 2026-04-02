import torch
import custom_ops_lib


# x_attention
def x_attention_npu(query, key_cache, value_cache, unshared_key, unshared_value, actual_shared_kvlen, decode_step, 
                    shared_block_tables = None, 
                    unshared_block_tables = None):
    return custom_ops_lib.x_attention(query, key_cache, value_cache, unshared_key, unshared_value, 
                                      shared_block_tables, unshared_block_tables, actual_shared_kvlen, decode_step)

# reshape cache kv
def cache_unshared_kv_npu(x_key_block, x_value_block, curr_key, curr_value, block_table, decode_step):
    return custom_ops_lib.cache_unshared_kv(x_key_block, x_value_block, curr_key, curr_value, block_table, decode_step)

# select unshared kv
def select_unshared_kv_npu(beam_index, x_key_block, x_value_block, block_table, group_token_num, decode_step, beam_size, layer_num):
    return custom_ops_lib.select_unshared_kv(beam_index, x_key_block, x_value_block, block_table, group_token_num, decode_step, beam_size, layer_num)

# beam search group
def beam_search_group_npu(log_probs, top_tokens, top_probs, sequence, current_step):
    return custom_ops_lib.beam_search_group(log_probs, top_tokens, top_probs, sequence, current_step)