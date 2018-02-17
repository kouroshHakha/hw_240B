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
    vn_in_max = 0.5 * (vsig ** 2) / snr_abs

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


            gmn = 8 * KT * gamma_n * (1 + gamma_p / gamma_n * vstar_n / vstar_p) * (f2 - f1) / vn_in_max
            Ibias = gmn * vstar_n

            input_scale = Ibias / 2 / input_db['ibias']
            load_scale = Ibias / 2 / load_db['ibias']
            gds_tot = input_db['gds'] * input_scale + load_db['gds'] * load_scale
            cdd_tot = input_db['cdd'] * input_scale + load_db['cdd'] * load_scale

            av = gmn / gds_tot
            bw = gds_tot / (cdd_tot + cload_min) / 2 / np.pi

            if av < av_min:
                continue


            if bw < bw_min:
                continue


            if Ibias < Ibias_min:
                output = dict()
                output['fg_tail'] = Ibias / tail_db['ibias']
                output['fg_input'] = Ibias / 2 / input_db['ibias']
                output['fg_load'] = Ibias / 2 / load_db['ibias']
                output['vb_load'] = vdd + load_db['vgs']
                output['vcm_in'] = vds_min + input_db['vgs']
                output['vb_tail'] = tail_db['vgs']
                output['lch'] = lch

                op_bias_point['output'] = output

                ss_params = dict(
                    input = input_db,
                    tail = tail_db,
                    load = load_db
                )
                op_bias_point['ss_params'] = ss_params

                specs = dict(
                    av=av,
                    Ibias=Ibias,
                    bw=bw,
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
        vsig=20e-3,
        vi_max=150e-3,
        snr_db=50,
        freq_range=(1.4e9, 1.6e9),
        cload_min=10e-15,
        av_min=8,
        bw_min=2e9
        )

    amp_specs = design_Simple_OTA_PSD(**specs)
    pprint.pprint(amp_specs)
    print('done')

if __name__ == '__main__':
    run_main()