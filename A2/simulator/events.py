"""
Adithya Bhaskar, 2023.
Implements events. Separated from the rest of the code so that adding 
functionality later won't conflict with other code.
"""

from config import *
from utils.globals import *

## Event statuses and times
EVENT_INIT = 30
EVENT_SCHEDULED = 31
EVENT_STARTED = 32
EVENT_FINISHED = 33
EVENT_CANCELED = 29

EVENT_NONE_TIME = SIM_END + 1
EVENT_NONE_ID = -100

def set_event_none_time(event_none_time):
    global EVENT_NONE_TIME
    EVENT_NONE_TIME = event_none_time
## -- Event statuses and times

## Trigger reasons
NOT_FIRED = 10
FIRE_CAUSE_SCHEDULED = 11
FIRE_CAUSE_TRIGGERED_BY_ANOTHER = 12
## -- Trigger reasons

## Number of events
NUM_EVENTS = 0

def get_num_events():
    global NUM_EVENTS
    return NUM_EVENTS

def get_new_event_number():
    global NUM_EVENTS
    NUM_EVENTS += 1
    return NUM_EVENTS
## -- Number of events

class Event:
    def __init__(self, event_handler, event_time=EVENT_NONE_TIME, \
        finish_with_start=True, handler_kwargs={}):
        self.handler = event_handler
        self.status = EVENT_INIT if event_time == EVENT_NONE_TIME else \
            EVENT_SCHEDULED
        self.schedule_time = event_time
        self.event_number = get_new_event_number()
        handler_kwargs['event_number'] = self.event_number
        self.saved_kwargs = handler_kwargs
        self.start_triggers = set()
        self.finish_triggers = set()
        self.finish_with_start = finish_with_start
        self.trigger_reason = NOT_FIRED
        self.saved_event_info = {}
    
    def __lt__(self, other):
        return self.event_number < other.event_number
    
    def add_to_event_info(self, addn_event_info):
        if DETAILED_EVENT_INFO_SIM and addn_event_info is not None:
            self.saved_event_info.update(addn_event_info)
    
    def add_start_trigger(self, dependent_event):
        self.start_triggers.add(dependent_event)
        
    def add_finish_trigger(self, dependent_event):
        self.finish_triggers.add(dependent_event)
        
    def schedule_event(self, event_time=None):
        self.status = EVENT_SCHEDULED
        if event_time is not None:
            self.schedule_time = event_time
            
    def cancel_event(self):
        self.status = EVENT_CANCELED
        self.schedule_time = EVENT_NONE_TIME
    
    def fire_event(self, trigger_reason=FIRE_CAUSE_SCHEDULED, event_info={}):
        if self.status == EVENT_CANCELED:
            return
        self.status = EVENT_STARTED
        self.trigger_reason = trigger_reason
        self.saved_event_info["start_time"] = get_global_time()
        self.saved_kwargs["passed_event_info"] = event_info
        self.add_to_event_info(self.handler(**self.saved_kwargs))
        for trigger in self.start_triggers:
            trigger.fire_event(trigger_reason=FIRE_CAUSE_TRIGGERED_BY_ANOTHER, \
                event_info=self.saved_event_info)
        if self.finish_with_start:
            self.finish_event()
        
    def finish_event(self):
        if self.status == EVENT_CANCELED:
            return
        self.status = EVENT_FINISHED
        self.saved_event_info["finish_time"] = get_global_time()
        for trigger in self.finish_triggers:
            trigger.fire_event(trigger_reason=FIRE_CAUSE_TRIGGERED_BY_ANOTHER, \
                event_info=self.saved_event_info)

def _test_event_impl():
    assert DETAILED_EVENT_INFO_SIM, "Need detailed info ON for testing!"
    def handler(**kwargs):
        print("Event {} fired at time {}.".format(kwargs['event_number'], \
            get_global_time()))
        if 'start_time' in kwargs['passed_event_info']:
            print("Triggering event started at time {}.".format(\
                kwargs['passed_event_info']['start_time']))
        if 'finish_time' in kwargs['passed_event_info']:
            print("Triggering event finished at time {}.".format(\
                kwargs['passed_event_info']['finish_time']))
        print("\n")
        return {}

    event1 = Event(handler, finish_with_start=False)
    event2 = Event(handler, finish_with_start=False)
    event3 = Event(handler, finish_with_start=True)
    event4 = Event(handler, finish_with_start=True)
    event5 = Event(handler, finish_with_start=False)
    
    event1.add_start_trigger(event2)
    event1.add_finish_trigger(event3)
    event2.add_start_trigger(event4)
    event2.add_finish_trigger(event5)
    
    ## At time 1
    # 1 fires, which triggers 2, which in turn triggers 4. The latter finishes.
    increment_global_time()
    event1.fire_event()
    
    ## At time 2
    # 2 finishes, which triggers 5 .
    increment_global_time()
    event2.finish_event()
    
    ## At time 3
    # 1 finishes, which triggers 3 as well as completes it.
    increment_global_time()
    event1.finish_event()
    
    ## At time 4
    # 5 finishes (no message printed)
    increment_global_time()
    event5.finish_event()

if __name__ == '__main__':
    _test_event_impl()