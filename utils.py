import json
import os
import tensorflow as tf
import numpy as np
import datetime as dt
from collections import defaultdict
import gaussian_variables as gv

def start_run():
    pid = os.getpid()
    run_id = "%s_%s" % (dt.datetime.now().strftime('%Y%m%d_%H%M%S'), pid)
    np.random.seed(0)
    tf.set_random_seed(0)

    print(''.join(['*' for _ in range(80)]))
    print('* RUN ID: %s ' % run_id)
    print(''.join(['*' for _ in range(80)]))
    return run_id

def get_hypers(args, default_hypers_path):
    # get the default
    with open(default_hypers_path, 'r') as f:
        hypers = json.load(f)
    # update according to --config-file
    config_file = args.get('--config-file')
    if config_file is not None:
        with open(config_file, 'r') as f:
            hypers.update(json.load(f))
    # update according to --config
    config = args.get('--config')
    if config is not None:
        hypers.update(json.loads(config))
    return hypers

def get_device_string(device_id):
    return '/cpu:0' if device_id < 0 else '/gpu:%s' % device_id

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess

def restrict_dataset_size(dataset, size_fraction):
    data_subset_size = int(np.floor(size_fraction * len(dataset[0])))
    return tuple([d[:data_subset_size] for d in dataset])

def batched(dataset, hypers):
    bs = hypers['batch_size']
    permute = False
    if permute:
        perm = np.random.permutation(range(len(dataset[0])))
        dataset[0] = np.array(dataset[0][perm])
        dataset[1] = np.array(dataset[1][perm])
    for batch_ptr in range(0, len(dataset[0]), bs):
        batch_ipts = dataset[0][batch_ptr:(batch_ptr+bs)]
        batch_opts = dataset[1][batch_ptr:(batch_ptr+bs)]
        yield batch_ipts, batch_opts

def make_optimizer(model_and_metrics, hypers):
    if hypers['optimizer'].strip().lower() == 'adam':
        optimizer = tf.train.AdamOptimizer(hypers['learning_rate'])
    elif hypers['optimizer'].strip().lower() == 'momentum':
        optimizer = tf.train.MomentumOptimizer(hypers['learning_rate'], 0.9)
    elif hypers['optimizer'].strip().lower() == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(hypers['learning_rate'])
    else:
        raise NotImplementedError('optimizer "%s" not recognized' % hypers['optimizer'])
        
    if hypers['gradient_clip'] > 0:
        gvs = optimizer.compute_gradients(model_and_metrics['metrics']['loss'])
        capped_gvs = [(tf.clip_by_value(grad, -hypers['gradient_clip'], hypers['gradient_clip']), var) 
                      for grad, var in gvs
                      if grad is not None]
        train_op = optimizer.apply_gradients(capped_gvs, global_step = model_and_metrics['global_step'])
    else:
        train_op = optimizer.minimize(model_and_metrics['metrics']['loss'],
                                      global_step = model_and_metrics['global_step'])
    return train_op

def update_prior_from_posterior(sess, model):
    if not hasattr(model, prior_update_assigners):
        assigners = []
        for p in model.parameters:
            assigners.append(tf.assign(p.prior.mean, p.value.mean))
            assigners.append(tf.assign(p.prior.var,  p.value.var ))
        model.prior_update_assigners = assigners
    sess.run(model.prior_update_assigners)

def run_one_epoch(sess, data, model, metrics, train_op, hypers, dynamic_hypers):
    learning_curve = []
    
    fetch_list = [metrics]
    if train_op is not None:
        fetch_list.append(train_op)
    
    feed_dict = {}
    if 'loss_n_samples' in hypers:
        feed_dict[model.placeholders['loss_n_samples']] = dynamic_hypers['loss_n_samples']
    
    count = 0
    running_accuracy = running_logprob = 0
    for batch in batched(data, hypers):
        feed_dict.update(
            {model.placeholders['ipt_mean']: batch[0],
            model.placeholders['target']:   batch[1]})
        result = sess.run(fetch_list, feed_dict)
        if 'prior_update' in hypers and hypers['prior_update']:
            update_prior_from_posterior(sess, model)
        new_count = count + len(batch[0])
        running_accuracy = \
            (count * running_accuracy + len(batch[0]) * result[0]['accuracy']) \
            / new_count
        running_logprob = \
            (count * running_logprob + len(batch[0]) * result[0]['logprob']) \
            / new_count
        count = new_count
        result[0].update({'count': len(batch[0]),
                          'running_accuracy': running_accuracy,
                          'running_logprob': running_logprob})
        learning_curve.append(result[0])
    return learning_curve

def train_valid_test(data, sess, model_and_metrics, train_op, hypers, verbose=True):
    train_op_dict = {'train': train_op, 'valid': None, 'test': None}
    summary = defaultdict(list)
    for section in hypers['sections_to_run']:
        dynamic_hypers = {h: hypers[h][section] 
                          for h in hypers 
                          if isinstance(hypers[h], dict) and section in hypers[h]}
        summary[section].append(run_one_epoch(
            sess, data[section], 
            model_and_metrics['model'], model_and_metrics['metrics'], 
            train_op_dict[section], hypers, dynamic_hypers
        ))
    accuracies = {}
    for section in hypers['sections_to_run']:
        accuracies[section] = summary[section][-1][-1]['running_accuracy']
        if verbose:
            print(' %s accuracy = %.4f | logprob = %.4f | KL term = %s' % (section, accuracies[section], 
                                                          summary[section][-1][-1]['running_logprob'],
                                                          summary[section][-1][-1]['all_surprise']/hypers['dataset_size']),
             end='')
    if verbose:
        print()
    return summary, accuracies

def piecewise_anneal(hypers, var_name, global_step):
    return hypers[var_name] * tf.clip_by_value((tf.to_float(global_step) - hypers['warmup_updates'][var_name])/hypers['anneal_updates'][var_name], 0.0, 1.0)


def get_predictions(data, sess, model, hypers):
    predictions = []
    output = model(model.placeholders['ipt_mean'])
    if isinstance(output, gv.GaussianVar):
        out_cov  =  tf.broadcast_to(output.var, [tf.shape(output.mean)[0], 
                                                                  tf.shape(output.mean)[1], 
                                                                  tf.shape(output.mean)[1]])
        out_mean = output.mean
    else:
        out_cov = tf.tile(tf.constant([[[0,0],[0,0]]]), [tf.shape(output)[0], 1,1])
        out_mean = output
    for batch in batched(data, hypers):
        result = sess.run({'mean':out_mean, 'cov':out_cov}, 
                          {model.placeholders['ipt_mean']: batch[0]})
        predictions.append((batch[0], result))
    
    x = np.concatenate([p[0] for p in predictions]).reshape(-1)
    y = {}
    for v in ['mean', 'cov']:
        y[v] = np.concatenate([p[1][v] for p in predictions])
    
    return (x,y)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (complex, np.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)