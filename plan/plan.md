Proposed Changes
v6.py
[MODIFY] 
v6.py
Update 
load_wagon_config
 to use:
Transporter Name (or Wagon Number)
Handle missing Lift Stroke Speed and No Of Station To Stop by providing default values.
Verification Plan
Automated Tests
Run the command:
python v6.py --input tanks_csv.csv --speeds "Wagon 1"
Verify that the sequence table is printed correctly.