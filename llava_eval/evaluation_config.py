from prompt import *
import json
from prompt.mvtec_2 import (
    MVTEC_0SHOT2_CNAME,
    MVTEC_0SHOT2_CONTEXT1,
    MVTEC_0SHOT2_CONTEXT2_NOHINT,  # 无输出格式限制
    MVTEC_0SHOT2_CONTEXT3_NOHINT,
    MVTEC_0SHOT2_APPEARANCE_NOHINT,  # 外观描述，无格式限制
    DETAILMODE2_QUESTION,  # 解释模式
)
from prompt.cotm_v3 import COTMV3_PROMPT
BASE_PROMPT="Are there any defects in this image?\nA. Yes\nB. No\nAnswer with the option's letter from the given choices directly."

SUPPORTED_QSET = {
    'mvtec': {
        'v0': MVTEC_PROMPT_V0, 
        'v1': MVTEC_PROMPT_V1_1, 
        'v2': MVTEC_PROMPT_V1, 
        'v3':MVTEC_PROMPT_V1_2,
        'v0_nh': MVTEC_PROMPT_V0_NOHINT,
        'v1_nh': MVTEC_PROMPT_V1_1_NOHINT,
        'v2_nh': MVTEC_PROMPT_V1_NOHINT,
        'v2_no':MVTEC_PROMPT_V1_NOOP,
        'explain_context': MVTEC_0SHOT2_CONTEXT3_NOHINT,
        'v3_nh': MVTEC_PROMPT_V1_2_NOHINT,
        'cotm_v3': COTMV3_PROMPT,
        'v0_role': MVTEC_PROMPT_V_ROLE,  
    }, 
    'pcb_bank': {
        'v0': BASE_PROMPT,
        'prod_v1': OTHER_DATASETS["pcb_bank"]["prompt_v1"],
    },
    'WFDD': {
        'v0': BASE_PROMPT,
        'prod_v1': OTHER_DATASETS["WFDD"]["prompt_v1"],
    },
    'mvtec_1shot': {
        'v0': MVTEC_PRODUCTION_1SHOT_V0,
        'v1': MVTEC_PRODUCTION_1SHOT_V1_1,
        'v2':MVTEC_PRODUCTION_1SHOT_V1,
        'v3':MVTEC_PRODUCTION_1SHOT_V1_2,
    },
    'mmad_mvtec':{},
}
