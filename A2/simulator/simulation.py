"""
Adithya Bhaskar, 2023.
This file exposes functions that simulate a general class of discrete events
indexed by time.
"""

from config import *
from utils.globals import *
from simulator.events import *

import os
from queue import PriorityQueue

## Simulation parameters
EVENT_QUEUE_MAX_SIZE = 10000000
## -- Simulation parameters

## Logging
def save_to_logs(log_dir, logs):
    log_output = []
    if "info" in logs:
        log_output.append("*** Information Log ***\n")
        for key in logs["info"]:
            if key != "general":
                log_output.append("{}".format(key))
            for entry in logs["info"][key]:
                log_output.append(entry)
        log_output.append("\n")
    
    if "metrics" in logs:
        log_output.append("*** Metrics ***\n")
        metrics = dict(sorted(logs["metrics"].items()))
        for key in metrics:
            value = metrics[key]
            log_output.append("{}\t:\t{}".format(key, value))
    
    with open(os.path.join(log_dir, "simulation.log"), 'w+') as f:
        f.write("\n".join(log_output))
    
    for key in logs:
        if key in ["info", "metrics"]:
            continue
        if type(logs[key]) == list:
            with open(os.path.join(log_dir, "{}.log".format(key)), 'w+') as f:
                f.write("\n".join(logs[key]))
        else:
            key_dir = os.path.join(log_dir, key)
            if not os.path.exists(key_dir):
                os.makedirs(key_dir, exist_ok=True)
            for sk in logs[key]:
                with open(os.path.join(key_dir, "{}.log".format(sk)), 'w+') as f:
                    f.write("\n".join(logs[key][sk]))
        
## -- Logging

class Simulation:
    def __init__(self, logging_disabled=False):
        self.event_queue = PriorityQueue(EVENT_QUEUE_MAX_SIZE)
        self.logs = {}
        self.canceled_events = set()
        self.sim_finished = False
        self.final_callback = None
        self.aggregated_metric_sum = {}
        self.aggregated_metric_square_sum = {}
        self.aggregated_metric_count = {}
        self.logging_disabled = logging_disabled
        if not logging_disabled:
            self.log_number = get_next_log_number()
            self.log_dir = os.path.join(LOG_DIR, "run{}".format(self.log_number))
            os.makedirs(self.log_dir, exist_ok=True)
    
    def set_final_callback(self, callback):
        self.final_callback = callback
        
    def get_log(self, key):
        return self.logs[key]

    def set_log(self, key, value):
        if not self.logging_disabled:
            self.logs[key] = value
        
    def log_info(self, info, key="general", log_level=1):
        if self.logging_disabled or log_level > VERBOSITY:
            return
        if "info" not in self.logs:
            self.logs["info"] = {}
        if key not in self.logs["info"]:
            self.logs["info"][key] = []
        self.logs["info"][key].append(info)
        
    def log_node(self, node, info, key="general", add_timestamp=True):
        if self.logging_disabled:
            return
        node_str = "node{}".format(node)
        if node_str not in self.logs:
            self.logs[node_str] = {}
        if key not in self.logs[node_str]:
            self.logs[node_str][key] = []
        if add_timestamp:
            info = "[{:.3f}] {}".format(ticks_to_seconds(get_global_time()), \
                info)
        self.logs[node_str][key].append(info)
    
    def log_metric(self, metric_name, metric_value):
        if self.logging_disabled:
            return
        if "metrics" not in self.logs:
            self.logs["metrics"] = {}
        self.logs["metrics"][metric_name] = metric_value
    
    def log_aggregate_metric(self, metric_name, metric_value):
        if metric_name not in self.aggregated_metric_count:
            self.aggregated_metric_count[metric_name] = 0
            self.aggregated_metric_sum[metric_name] = 0
            self.aggregated_metric_square_sum[metric_name] = 0
        self.aggregated_metric_count[metric_name] += 1
        self.aggregated_metric_sum[metric_name] += metric_value
        self.aggregated_metric_square_sum[metric_name] += metric_value**2
        
    def schedule_event_at_time(self, event: Event, time: int):
        assert self.event_queue.qsize() < EVENT_QUEUE_MAX_SIZE, \
            "Simulation: event_queue overflowing at {} ticks!".format(\
            get_global_time())
        event.schedule_event(time)
        self.event_queue.put((time, event))
    
    def schedule_multiple_events_at_times(self, events, times):
        for event, time in zip(events, times):
            self.schedule_event_at_time(event, time)
    
    def cancel_event(self, event_number):
        self.canceled_events.add(event_number)
        
    def pop_and_execute_next_event(self):
        PRINT_INTERVAL = 100000
        event_time, next_event = self.event_queue.get()
        gtime = get_global_time()
        # print(event_time)
        # if gtime // PRINT_INTERVAL != event_time // PRINT_INTERVAL and VERBOSITY >= 2:
        #     print("{} ticks done.".format(PRINT_INTERVAL*(event_time//PRINT_INTERVAL)))
        assert event_time >= gtime, "Simulation: Next event "+\
            "is in the past!"
        if event_time > SIM_END:
            print("Simulation: Warning: Ending simulation at {} ".format(SIM_END)+\
                "even though an event scheduled at {}".format(event_time))
            self.sim_finished = True
            return
        event_time = set_and_return_global_time(event_time)
        if next_event.status == EVENT_CANCELED or next_event.event_number in \
            self.canceled_events:
            # Canceled, skip it ; Canceling is idempotent
            next_event.cancel_event()
            self.canceled_events.remove(next_event.event_number)
        elif next_event.status == EVENT_STARTED:
            # Event should finish now
            next_event.finish_event()
        elif next_event.status == EVENT_SCHEDULED:
            # Event should start now
            next_event.fire_event()
        else:
            # Event somehow finished, skip it.
            assert next_event.status == EVENT_FINISHED, "Simulation: Event still "+\
                "in INIT stage but about to fire!"
    
    def run_simulation(self):
        set_event_none_time(SIM_END+1)
        while self.event_queue.qsize() > 0:
            if self.sim_finished:
                break
            self.pop_and_execute_next_event()
        self.sim_finished = True
        set_and_return_global_time(SIM_END)
        if self.final_callback is not None:
            self.final_callback(log_dir=self.log_dir)
        for metric_name in self.aggregated_metric_count:
            metric_mean = self.aggregated_metric_sum[metric_name] / \
                self.aggregated_metric_count[metric_name]
            metric_square_mean = self.aggregated_metric_square_sum[metric_name] / \
                self.aggregated_metric_count[metric_name]
            metric_std = (metric_square_mean-(metric_mean**2))**0.5
            metric_str = "{:.4f} +/- {:.4f}".format(metric_mean, metric_std)
            self.log_metric(metric_name, metric_str)
        if not self.logging_disabled:
            save_to_logs(self.log_dir, self.logs)

SIMULATION = Simulation()

if __name__ == '__main__':
    pass