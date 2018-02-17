# -*- coding: utf-8 -*-

import pprint

import numpy as np
import scipy.optimize as sciopt

from bag.util.search import BinaryIterator
from verification_ec.mos.query import MOSDBDiscrete


def get_db(spec_file, lch, intent, interp_method='spline', sim_env='tt'):
    # initialize transistor database from simulation data
    mos_db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    # set process corners
    mos_db.env_list = [sim_env]
    # set layout parameters
    mos_db.set_dsn_params(intent=intent, lch=lch)
    return mos_db


def design_Simple_OTA_PSD(nmos_spec_file, pmos_spec_file, sim_env, lch_list,
                          vdd, vo_cm, vds_min,
                          vsig, vi_max, snr_db, freq_range,
                          cload_min, av_min, bw_min):
    """input: vdd, vo_cm, vds_min, vi_max, av_min, bw_min, vsig, snr, cload_min, [f1,f2], integral=False
    output: lch, vb_load, fg_load, vb_tail, fg_tail, vcm_in, fg_input, op_bias_point
    Output structure:
    op_bias_point = {
    	'output':{
    		'fg_tail': None
    		'fg_input': None
    		'fg_load': None
    		'vb_load': None
    		'vcm_in': None
    		'vb_tail': None
    		'lch': None
    	}

    	'ss_params':{
    		'input': None
    		'load': None
    		'tail': None
    	}

    	'specs':{
    		'Ibias': None
    		'av': None
    		'bw': None
    		'snr': None
    	}

    }
    """

    KT = 4e-21
    Ibias_min = np.float('inf')

    # algorithm

    vstar_n = vi_max
    f1, f2 = freq_range

    snr_abs = 10 ** (snr_db / 10)

    op_bias_point = dict()
    for lch in lch_list:
        db_nmos = get_db(nmos_spec_file, lch=lch, intent='lvt', interp_method='spline', sim_env=sim_env)
        db_pmos = get_db(pmos_spec_file, lch=lch, intent='lvt', interp_method='spline', sim_env=sim_env)

        # for now vstar is swept in vgs range but needs to be more accurate
        vstar_res = 20e-3
        # vstar_fun = db_pmos.get_function('vstar', sim_env=sim_env)
        # vstar_p_min = fmin(vstar_fun, np.array([0,0.1,0.1]))
        vstar_p_min, vstar_p_max = (vstar_n, 9.5 * vstar_n)
        num_vstar_p = int(np.ceil((vstar_p_max - vstar_p_min) / vstar_res)) + 1
        vstar_p_vec = np.linspace(vstar_p_min, vstar_p_max, num_vstar_p, endpoint=True)
        print (vstar_p_vec)

        for vstar_p in vstar_p_vec:
            input_db = db_nmos.query(vds=vo_cm - vds_min, vbs=-vds_min, vstar=vstar_n)
            tail_db = db_nmos.query(vds=vds_min, vbs=0, vstar=vstar_n)
            load_db = db_pmos.query(vds=vo_cm - vdd, vbs=0, vstar=vstar_p)

            gamma_n = input_db['gamma']
            gamma_p = load_db['gamma']

            id_input_finger = input_db['ibias']
            id_load_finger = load_db['ibias']
            id_tail_finger = tail_db['ibias']
            ratio_load_to_input = id_input_finger / id_load_finger
            ratio_tail_to_input = 2 * id_input_finger / id_tail_finger

            cdd_unit = input_db['cdd'] + ratio_load_to_input * load_db['cdd']

            gds_unit = input_db['gds'] + ratio_load_to_input * load_db['gds']
            gm_unit = input_db['gm']

            av_unit = gm_unit / gds_unit

            if av_unit < av_min:
                print ("av", av_unit)
                continue

            vn_out_max = 0.5 * (av_unit * vsig) ** 2 / snr_abs
            delta = (np.arctan(f2 / bw_min) - np.arctan(f1 / bw_min)) / 2 / np.pi
            print ("delta", delta)
            cout = av_unit * 8 * KT * gamma_n * (1+gamma_p / gamma_n * vstar_n / vstar_p) * delta / vn_out_max
            print ("cout",cout)
            c_ext = 0
            if (cout < cload_min):
                print ("cload is big enough to handle noise")
                max_gain_bw_unit = gm_unit / 2 / np.pi / cdd_unit
                gmn = av_unit * 2 * np.pi * bw_min * cload_min / (1 - av_unit * bw_min / max_gain_bw_unit)
                scale = gmn / gm_unit
            else:
                print ("c_ext is needed")
                gmn = av_unit * 2 * np.pi * bw_min * cout
                scale = gmn / gm_unit
                print ("total c", cload_min + scale * cdd_unit)
            if (cout < cload_min + scale * cdd_unit):
                max_gain_bw_unit = gm_unit / 2 / np.pi / cdd_unit
                print ("max_gain_bw_unit", max_gain_bw_unit)
                print ("GBW",  av_unit * bw_min)
                gmn = av_unit * 2 * np.pi * bw_min * cload_min / (1 - av_unit * bw_min / max_gain_bw_unit)
                print ("gmn", gmn)
                scale = gmn / gm_unit
            else:
                c_ext = cout - (cload_min + scale * cdd_unit)

            if (scale < 0):
                # maximum achievable gain bandwidth is violated
                print ("design not feasible")
                continue

            print ("c_ext", c_ext)
            Ibias = gmn * vstar_n

            gds_amp = gds_unit * scale
            cdd_amp = cload_min + scale * cdd_unit + c_ext

            bw_amp = gds_amp / cdd_amp / 2 / np.pi

            if bw_amp < bw_min:
                print ("bw_amp",bw_amp)
                continue


            if Ibias < Ibias_min:
                output = dict()
                output['fg_tail'] = scale * ratio_tail_to_input
                output['fg_input'] = scale
                output['fg_load'] = scale * ratio_load_to_input
                output['vb_load'] = vdd + load_db['vgs']
                output['vcm_in'] = vds_min + input_db['vgs']
                output['vb_tail'] = tail_db['vgs']
                output['lch'] = lch
                output['c_ext'] = c_ext

                op_bias_point['output'] = output

                ss_params = dict(
                    input = input_db,
                    tail = tail_db,
                    load = load_db
                )
                op_bias_point['ss_params'] = ss_params

                specs = dict(
                    av=av_unit,
                    Ibias=Ibias,
                    bw=bw_amp,
                    snr=snr_db)
                op_bias_point['specs'] = specs

                Ibias_min = Ibias

        if Ibias_min == np.float('inf'):
            print ("sry!! your design was gain/bw limited. please apply gain/bw limited methodology")
        else: return op_bias_point

def run_main():
    interp_method = 'spline'
    sim_env = 'tt'
    nmos_spec = 'specs_mos_char/nch_w0d5.yaml'
    pmos_spec = 'specs_mos_char/pch_w0d5.yaml'
    intent = 'lvt'

    specs = dict(
        nmos_spec_file=nmos_spec,
        pmos_spec_file=pmos_spec,
        lch_list = [90e-9],
        sim_env=sim_env,
        vdd=1.2,
        vo_cm=0.6,
        vds_min=0.2,
        vsig=50e-3,
        vi_max=150e-3,
        snr_db=50,
        freq_range=(1e6, 100e9),
        cload_min=10e-15,
        av_min=10,
        bw_min=1e9
        )

    amp_specs = design_Simple_OTA_PSD(**specs)
    pprint.pprint(amp_specs)
    print('done')

if __name__ == '__main__':
    run_main()