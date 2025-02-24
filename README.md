
# decat-obsplan-generator

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/decat-obsplan-generator?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/decat-obsplan-generator/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/kdesoto-astro/decat-obsplan-generator/smoke-test.yml)](https://github.com/kdesoto-astro/decat-obsplan-generator/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/kdesoto-astro/decat-obsplan-generator/branch/main/graph/badge.svg)](https://codecov.io/gh/kdesoto-astro/decat-obsplan-generator)
[![Read The Docs](https://img.shields.io/readthedocs/decat-obsplan-generator)](https://decat-obsplan-generator.readthedocs.io/)

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

A repository badge was added to show that this project uses the python-project-template, however it's up to
you whether or not you'd like to display it!

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

## How to generate schedules

This repository connects to decat-pointings to automatically generate nightly obsplans.
After installing locally via git clone or fork:

* First, you want to navigate to decat-pointings within this directory. Then update your version of the subdirectory using
>> cd decat-pointings
>> git pull
* Next, to generate an optimized observing plan, navigate to src/decat_obsplan_generator, and run the scheduling.py file:
>> cd src/decat_obsplan_generator
>> python scheduling.py
* Press enter to generate schedule for that same night, or change for other nights. An optimized schedule will automatically be generated, displayed, and saved in the nightly folder in decat-pointings.
* If this schedule looks good, simply navigate back to decat-pointings, git add, commit, and push the changes!
* If you want to edit this schedule, you can either: (1) modify the order numbers in decat-pointings/{date}/{date}_all_targets.csv. Running scheduling.py after will automatically read in the modified table and generated updated obsplans. (2) use the inline commands (detailed below) to augment the schedule.
* The table, obsplan, airmass plot, and moon distance plots are all generated and saved in the decat-pointings folder after each edit.

## Inline commands (after calling scheduling.py)

These commands can be called after "python scheduling.py" brings up the schedule display.

* move {target_name} {order} : moves the target with name target-name to specific order. Adjusts the later targets accordingly.
* add {target_name} : add target to the schedule. This will try to insert target in the spot that requires the least slewing.
* remove {target_name} : this will remove the target from the current schedule.
* swap {target1} {target2} : swap two targets in observing order.
* optimize : this will generate an optimized schedule based on coordinates and slew times. Priorities are ignored!
* optimize-prio : same as optimize, but it priorities objects of higher priority first.
* help : this will show all valid commands
