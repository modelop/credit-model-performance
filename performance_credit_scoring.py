#from modelop.monitors.assertions import check_input_types
from pathlib import Path
import modelop.schema.infer as infer
import pandas as pd
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

PROB_COLUMM = []
LABEL_COLUMN = []

# modelop.init
def init(init_param):

    global PROB_COLUMM
    global ACTUAL_COLUMN

    job_json = init_param


    if job_json is not None:
        logger.info(
            "Parameter 'job_json' is present and will be used to extract "
            "'label_column' and 'score_column'."
        )
        input_schema_definition = infer.extract_input_schema(job_json)
        monitoring_parameters = infer.set_monitoring_parameters(
            schema_json=input_schema_definition, check_schema=True
        )
        PROB_COLUMM = monitoring_parameters['score_column']
        ACTUAL_COLUMN = monitoring_parameters['label_column']

    else:
        logger.info(
            "Parameter 'job_json' it not present, attempting to use "
            "'label_column' and 'score_column' instead."
        )
        if LABEL_COLUMN is None:
            missing_args_error = (
                "Parameter 'job_json' is not present,"
                " but 'label_column'. "
                "'label_column' input parameter is"
                " required if 'job_json' is not provided."
            )
            logger.error(missing_args_error)
            raise Exception(missing_args_error)


# modelop.metrics
def metrics(data: pd.DataFrame) -> dict:
    giniNormalized = gini_normalized(data[ACTUAL_COLUMN],data[PROB_COLUMM])


    return {'Gini' :
        [{
            'test_name': "Gini",
            'test_category': "gini",
            'test_type': "gini",
            'test_id': "gini_test",
            'values': { "Normalized Gini": giniNormalized }
        }],
        'Normalized_Gini': giniNormalized
    }


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    allData = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    allData = allData[ np.lexsort((allData[:,2], -1* allData[:,1])) ]
    totalLosses = allData[:,0].sum()
    giniSum = allData[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def main():
    raw_json = Path('Gini/example_job.json').read_text()
    init_param = {'rawJson': raw_json}
    init(init_param)
    print('initialized parameters from job_json.')
    print(PROB_COLUMM)
    print(ACTUAL_COLUMN)
    data = pd.read_csv('Gini/rob_test.csv')
    print('read data.')
    result = metrics(data)
    print(json.dumps(result, indent=3, sort_keys=True))
    print('done.')

if __name__ == '__main__':
    main()