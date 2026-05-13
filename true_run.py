'''
This script runs the TRUE model and generates the synthetic data for the TRUE model.
'''

__author__ = 'Mathias Methlie Nilsen'

import jutuldarcy as jd
import pandas as pd
import datetime as dt
import numpy as np

datatypes = [
    'WOPR:P1',
    'WOPR:P2',
    'WOPR:P3',
    'WOPR:P4',
    'WGPR:P1',
    'WGPR:P2',
    'WGPR:P3',
    'WGPR:P4',
    'WWPR:P1',
    'WWPR:P2',
    'WWPR:P3',
    'WWPR:P4',
]

# Dates
startdate = dt.datetime(2000, 1, 1)
rep_dates  = pd.date_range(
    dt.datetime(2000, 2, 1), 
    dt.datetime(2009, 12, 1), 
    freq='MS'
).to_pydatetime().tolist()

idx = np.linspace(6, len(rep_dates)-6, 5, dtype=int)
assim_dates = [rep_dates[i] for i in idx]


if __name__ == '__main__':

    # Save dates
    with open('TRUE_MODEL/report_dates.csv', 'w') as f:
        for d in [d.strftime('%Y-%m-%d %H:%M:%S') for d in rep_dates]:
            f.write(f'{d}\n')

    with open('DATA/assim_dates.csv', 'w') as f:
        for d in [d.strftime('%Y-%m-%d %H:%M:%S') for d in assim_dates]:
            f.write(f'{d}\n')


    # Simulate the TRUE model
    res = jd.simulate_data_file('TRUE_MODEL/TRUE_MODEL.DATA', convert=True)

    # Extract results
    data = {}
    results = {}

    for datatype in datatypes:
        key, well = datatype.split(':')
        data_values = []
        res_values = []
        for d, day in enumerate(res['DAYS']):
            date = startdate + dt.timedelta(days=int(day))
            if date in assim_dates:
                data_values.append(res['WELLS'][well][key][d])
            if date in rep_dates:
                res_values.append(res['WELLS'][well][key][d])

        data[datatype] = data_values
        results[datatype] = res_values

    # Save data and results
    results = pd.DataFrame(results, index=rep_dates)
    results.index.name = 'date'
    results.to_pickle('TRUE_MODEL/true_run.pkl')
    print(results)

    data = pd.DataFrame(data, index=assim_dates)
    data.index.name = 'date'

    data.to_pickle('./DATA/true_data.pkl')

    data_var = pd.DataFrame(columns=data.columns, index=data.index)
    data_var.index.name = 'date'

    # Set units and variance of data as attributes
    std = 0.05 # 5% error
    for col in data.columns:
        var = (std * data[col].values) ** 2
        # Use the median across all dates as a single shared value,
        # so every observation for this column has identical noise.
        var_scalar = float(np.percentile(var, 75))
        var = np.full(len(data), var_scalar)
        data.attrs[col] = {
            'unit': 'Sm3/DAY',
            'variance': var}

        _var = [['abs', float(v)] for v in var]
        data_var[col] = _var
        
    # Save to data
    data_var.to_pickle('./DATA/true_data_var.pkl')

        
