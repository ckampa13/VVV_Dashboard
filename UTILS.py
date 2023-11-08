# "local" imports
from CONFIG import EFTAnalysisDir
# from plotting import config_plots, get_label, ticks_in
# config_plots()
# from EFTAnalysis
import sys
sys.path.append(EFTAnalysisDir+'EFTAnalysisFitting/scripts/')
#from DATACARD_DICT import datacard_dict
#from CONFIG_VERSIONS import versions_dict, WC_ALL
from CONFIG_VERSIONS import WC_ALL
#from MISC_CONFIGS import template_filename_yields

def classify_histogram_keys(keys):
    bkgs = []
    systs = []
    datas = []
    for k in keys:
        is_bkg = True
        is_syst = False
        is_data = False
        if ('Up' in k) or ('Down' in k):
            is_syst = True
        has_WC = False
        for WC in WC_ALL:
            if WC in k:
                has_WC = True
                break
        if has_WC or ("sm" in k):
            is_bkg = False
        # check for data
        if "data_obs" in k:
            is_bkg = False
            is_syst = False
            is_data = True
        bkgs.append(is_bkg)
        systs.append(is_syst)
        datas.append(is_data)
    #return np.array(bkgs), np.array(systs)
    return bkgs, systs, datas
