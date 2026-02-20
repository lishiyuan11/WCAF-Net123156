# evaluate max F-measure, S-measure and MAE

import os
from evaluator.evaluator import evaluate_dataset
from evaluator.utils import write_doc
from libs.log import create_logger
import config


def evaluate(roots, doc_path, num_thread, pin):
    datasets = roots.keys()
    for dataset in datasets:
        # Evaluate predictions of "dataset".
        results = evaluate_dataset(roots=roots[dataset], 
                                   dataset=dataset,
                                   batch_size=1, 
                                   num_thread=num_thread, 
                                   demical=True,
                                   suffixes={'gt': '.png', 'pred': '.png'},
                                   pin=pin)

        # Save evaluation results.
        content = '{}:\n'.format(dataset)
        #content += 'max-Fmeasure={}'.format(results['max_f'])
        content += 'max-Fmeasure={} mean-Fmeasure={} '.format(results['max_f'], results['mean_f'])
        content += 'max-Emeasure={} mean-Emeasure={} '.format(results['max_e'], results['mean_e'])
        content += ' '
        content += 'Smeasure={}'.format(results['s'])
        content += ' '
        content += 'MAE={}\n'.format(results['mae'])
        write_doc(doc_path, content)

    return content


# ------------- end -------------

if __name__ == "__main__":
    logger = create_logger('evaluate')

    eval_device = '0'
    eval_doc_path = './eva.txt'
    eval_num_thread = 10

    logger.info('eval_doc_path:{}\neval_num_thread:{}'.format(eval_doc_path, eval_num_thread))
    logger.info('model name:{}'.format(config.model_name))

    # An example to build "eval_roots".
    eval_roots = dict()

    for dataset in config.test_datasets:
        print(dataset)
        logger.info(dataset)
        roots = {'gt': config.test_path + '/' + dataset + '/gt/',
                'pred': config.prediction_save_path + '/' + dataset + '/'}
        eval_roots[dataset] = roots
        os.environ['CUDA_VISIBLE_DEVICES'] = eval_device
        content = evaluate(roots=eval_roots, 
                doc_path=eval_doc_path,
                num_thread=eval_num_thread,
                pin=False)

        logger.info('pred:{}'.format(roots['pred']))

        logger.info(content)
        print(content)



