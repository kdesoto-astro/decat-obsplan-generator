# Where schedule and observing blocks are generated.
from astroplan.scheduling import ObservingBlock, Scheduler, Schedule
import astropy.units as u
from .helpers import ctio_location

def generate_observing_block(
    name, ra, dec, duration,
    night_start, night_end, priority=1,
):
    """Generate astroplan observing block
    """
    start_time, end_time = observable_range(ra, dec, night_start, night_end)
    return DECATObservingBlock(
        name, duration, start_time, end_time,
        priority, program
    )

class DECATObservingBlock:
    def __init__(name, duration, start_time, end_time, priority, program):
        self.name = name
        self.duration = duration
        self.earliest_time = start_time
        self.latest_time = end_time
        self.priority = priority
        self.program = program


class DECATSchedule(Schedule):
    """Modified schedule for DECAT, which includes total program
    allocated times.
    """
    def __init__(self, start_time, end_time):
        """Adds program time dictionary."""
        super().__init__(start_time, end_time)
        self.program_time_fn = ""
        self.program_times = {} # TODO: populate


class DECATScheduler(Scheduler):
    def __init__(self, observer=None):
        if observer is None:
            observer = ctio_location()
        transitioner = Transitioner(slew_rate=5/11. * u.deg/u.s)
        super().__init__(constraints=[], observer=observer, transitioner=transitioner)

    def _make_schedule(self, blocks):
        """Make schedule with slightly smarter logic:
        - For each program, keep track of cumul. time during night (as frac of total alloc)
        - Each target prio = combining program prio + size of observation window (num blocks)
        - For current start block, rate prio of compatible targets --> add highest prio
        - Refresh prios, continue for rest of schedule
        """
        cumul_program_times = {
            k: 0 for (k, v) in self.schedule.program_times.items()
        }
        current_time = self.schedule.start_time
        prios = np.zeros(len(blocks))
        num_blocks = np.zeros(len(blocks))
        total_blocks = (self.schedule.end_time - self.schedule.start_time) // self.time_resolution

        for i, b in enumerate(blocks):
            b.observer = self.observer
            start_time_window = (b.latest_time - duration) - b.earliest_time
            num_blocks[i] = start_time_window / time_resolution

        while (len(blocks) > 0) and (current_time < self.schedule.end_time):
            block_transitions = []
            for i, b in enumerate(blocks):
                # calculate transition time
                if len(self.schedule.observing_blocks) > 0:
                    trans = self.transitioner(
                        self.schedule.observing_blocks[-1], b, current_time, self.observer
                    )
                    trans.components = {
                        "slew_time": max(0 * u.s, trans.components['slew_time'] - 9 * u.s),
                        "readout_time": 29 * u.sec,
                    }
                else:
                    trans = TransitionBlock(
                        components = {'readout_time': 29 * u.sec},
                        start_time = current_time
                    )
                    
                transition_time = trans.duration
                block_transitions.append(trans)

                if current_time + transition_time < b.earliest_time:
                    prios[i] = 9999
                elif current_time + transition_time + duration > b.latest_time:
                    prios[i] = 9999
                else:
                    program_time_frac = cumul_program_times / self.schedule.program_times[b.program]
                    prios[i] = 10 * num_blocks / total_blocks + program_time_frac + 100 * self.priority

                best_idx = np.argmin(prios)

                if prios[best_idx] == 9999:
                    current_time += self.gap_time # no good target, add gap

                trans = block_transitions.pop(best_idx)

                if trans is not None:
                    self.schedule.insert_slot(trans.start_time, trans)
                    current_time += trans.duration

                # now assign the block itself times and add it to the schedule
                newb = blocks.pop(best_idx)
                newb.start_time = current_time
                current_time += newb.duration
                newb.end_time = current_time

                self.schedule.insert_slot(newb.start_time, newb)

        return self.schedule