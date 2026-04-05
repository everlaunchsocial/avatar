import numpy as np

with open('/workspace/HunyuanVideo-Avatar/hymm_sp/modules/models_audio.py', 'r') as f:
    content = f.read()

init_code = '''
    # TeaCache attributes
    enable_teacache = False
    teacache_cnt = 0
    teacache_num_steps = 20
    teacache_thresh = 0.15
    teacache_accumulated_distance = 0
    teacache_previous_modulated_input = None
    teacache_previous_residual = None
    teacache_skipped_steps = 0

'''

content = content.replace(
    '    def forward(\n        self,\n        x: torch.Tensor,\n        t: torch.Tensor, # Should be in range(0, 1000).',
    init_code + '    def forward(\n        self,\n        x: torch.Tensor,\n        t: torch.Tensor, # Should be in range(0, 1000).'
)

old_block_start = '''        # --------------------- Pass through DiT blocks ------------------------
        if not is_cache:
            for layer_num, block in enumerate(self.double_blocks):'''

new_block_start = '''        # --------------------- TeaCache Logic ------------------------
        should_calc = True
        if self.enable_teacache:
            modulated_inp = self.double_blocks[0].img_mod(vec)
            if hasattr(modulated_inp, 'shift'):
                modulated_inp_val = modulated_inp.shift
            elif isinstance(modulated_inp, tuple):
                modulated_inp_val = modulated_inp[0] if len(modulated_inp) > 0 else vec
            else:
                modulated_inp_val = modulated_inp
            
            if self.teacache_previous_modulated_input is not None and self.teacache_cnt > 0 and self.teacache_cnt < self.teacache_num_steps - 1:
                distance = (modulated_inp_val - self.teacache_previous_modulated_input).abs().mean()
                prev_mean = self.teacache_previous_modulated_input.abs().mean()
                if prev_mean > 0:
                    rel_distance = distance / prev_mean
                else:
                    rel_distance = distance
                
                coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                rescale_func = np.poly1d(coefficients)
                scaled_distance = rescale_func(rel_distance.item())
                self.teacache_accumulated_distance += scaled_distance
                
                if self.teacache_accumulated_distance < self.teacache_thresh:
                    should_calc = False
                    self.teacache_skipped_steps += 1
                else:
                    self.teacache_accumulated_distance = 0
            
            self.teacache_previous_modulated_input = modulated_inp_val.clone()
            self.teacache_cnt += 1
        
        # --------------------- Pass through DiT blocks ------------------------
        if not should_calc and self.teacache_previous_residual is not None:
            img = self.teacache_previous_residual
            img = self.final_layer(img, vec)
            img = self.unpatchify(img, tt, th, tw)
            if return_dict:
                out['x'] = img
                return out
            return img
        
        if should_calc and not is_cache:
            self._ori_img = img.clone() if self.enable_teacache else None
            for layer_num, block in enumerate(self.double_blocks):'''

content = content.replace(old_block_start, new_block_start)

old_extract = '''        img = x[:, :-txt_seq_len, ...]

        if get_sequence_parallel_state():
            img = all_gather(img, dim=1) 
        img = img[:, ref_length:]'''

new_extract = '''        img = x[:, :-txt_seq_len, ...]

        if get_sequence_parallel_state():
            img = all_gather(img, dim=1) 
        img = img[:, ref_length:]
        
        if self.enable_teacache and should_calc and hasattr(self, "_ori_img") and self._ori_img is not None:
            self.teacache_previous_residual = img.clone()
            self._ori_img = None'''

content = content.replace(old_extract, new_extract)

with open('/workspace/HunyuanVideo-Avatar/hymm_sp/modules/models_audio.py', 'w') as f:
    f.write(content)

print("TeaCache patch applied successfully!")
