with open('/workspace/HunyuanVideo-Avatar/hymm_sp/sample_gpu_poor.py', 'r') as f:
    content = f.read()

old = '    args = hunyuan_video_sampler.args'
new = '''    tc_model = hunyuan_video_sampler.model
    tc_model.enable_teacache = True
    tc_model.teacache_num_steps = args.infer_steps
    tc_model.teacache_thresh = 0.15
    tc_model.teacache_cnt = 0
    tc_model.teacache_accumulated_distance = 0
    tc_model.teacache_previous_modulated_input = None
    tc_model.teacache_previous_residual = None
    tc_model.teacache_skipped_steps = 0
    print(f"TeaCache ENABLED: thresh={tc_model.teacache_thresh}, steps={tc_model.teacache_num_steps}")
    
    args = hunyuan_video_sampler.args'''

content = content.replace(old, new)

with open('/workspace/HunyuanVideo-Avatar/hymm_sp/sample_gpu_poor.py', 'w') as f:
    f.write(content)

print("TeaCache activation added")
