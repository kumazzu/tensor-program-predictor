# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument, no-self-use, invalid-name
"""Base class of tuner"""
import logging
import matplotlib.pylab as plt
import numpy as np
from tvm.autotvm import feature

from ..measure import MeasureInput, create_measure_batch

from ..env import GLOBAL_SCOPE
import pickle
import time

logger = logging.getLogger("autotvm")


class Tuner(object):
    """Base class for tuners

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task
    """

    def __init__(self, task, **kwargs):
        self.param = kwargs
        self.recorder = None

        self.task = task

        # keep the current best
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None

    def has_next(self):
        """Whether has next untried config in the space

        Returns
        -------
        has_next: bool
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """get the next batch of configs to be measure on real hardware

        Parameters
        ----------
        batch_size: int
            The size of the batch

        Returns
        -------
        a batch of configs
        """
        raise NotImplementedError()

    def update(self, inputs, results):
        """Update parameters of the tuner according to measurement results

        Parameters
        ----------
        inputs: Array of autotvm.measure.MeasureInput
            The input for measurement
        results: Array of autotvm.measure.MeasureResult
            result for measurement
        """

    def exploration(self, plan_size):
        """Update parameters of the tuner according to measurement results

        Parameters
        ----------
        inputs: Array of autotvm.measure.MeasureInput
            The input for measurement
        results: Array of autotvm.measure.MeasureResult
            result for measurement
        """

    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), opt=None):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        """

        # self.exploration()
        self.opt = opt
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, "n_parallel", 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0

        self.e_i = []
        self.e_s = []
        self.xxs = []
        self.yys = []
        self.features = []

        while i < n_trial:
            if not self.has_next():
                break
            configs = self.next_batch(min(n_parallel, n_trial - i))
            mea_start = time.time() # 统计硬件测量用时
            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)
            mea_end = time.time() # END 统计硬件测量用时
            print(f"HW measurements cost: {mea_end-mea_start} s")

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    flops = 0
                    error_ct += 1

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

            i += len(results)
            for callback in callbacks:
                callback(self, inputs, results)

            ud_start = time.time() # 统计更新用时
            self.update(inputs, results)
            ud_end = time.time() # END 统计更新用时
            print(f"model update cost: {ud_end-ud_start} s")
            flops_max = -np.inf

            for inp, res in zip(inputs, results):
                # calculate flops max
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    if flops > flops_max:
                        flops_max = flops

        GLOBAL_SCOPE.in_tuning = False

        del measure_batch

    def tensor_program_predictor(self, n_trial, measure_option, early_stopping=None, callbacks=(), opt=None):
        p_i, p_s = self.exploration()
        self.opt = opt
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, "n_parallel", 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        old_level = logger.level
        GLOBAL_SCOPE.in_tuning = True
        offset = 100
        i = error_ct = 0
        self.e_i = []
        self.e_s = []
        self.xxs = []
        self.yys = []
        self.est = []

        for inp, res in zip(p_i, p_s):
            self.e_i.append(inp)
            self.e_s.append(res)

        while i < n_trial:
            if not self.has_next():
                break
            configs = self.next_batch(min(n_parallel, n_trial - i))
            mea_start = time.time() # 统计硬件测量用时
            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)
            mea_end = time.time() # END 统计硬件测量用时
            print(f"HW measurements cost: {mea_end-mea_start} s")

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    flops = 0
                    error_ct += 1

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

            i += len(results)
            for callback in callbacks:
                callback(self, inputs, results)

            flops_max = -np.inf

            for inp, res in zip(inputs, results):
                # calculate flops max
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    if flops > flops_max:
                        flops_max = flops

            for inp, res in zip(inputs, results):
                index = inp.config.index
                if res.error_no == 0:
                    try:
                        iid = p_i.index(index)
                        self.est.append(p_i[iid])
                        self.xxs.append(index)
                        flops = inp.task.flop / np.mean(res.costs)
                        self.yys.append(flops)
                    except:
                        pass

        GLOBAL_SCOPE.in_tuning = False
        for i, v, e in zip(self.xxs, self.yys, self.est):
            np.save(f"{self.opt.save_dir}/{self.opt.task}/output_{self.opt.trial}_{i}.npy", v)
            np.save(f"{self.opt.save_dir}/{self.opt.task}/estmated_{self.opt.trial}_{i}.npy", e)

            self.opt.trial += 1

        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        """load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (MeasureInput, MeasureResult) pair
            Previous tuning records
        """
        raise NotImplementedError()
