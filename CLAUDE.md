>> PLAN MODE: 
    >> create a TODO list when working on complex tasks to track progress and remain on track
    >> do not be YES MAN >> do WEBSEARCH >> give me PROS and CONS of atleast 3  approaches 

>> INSTALLATION
    >> in virtual environment {venv_<project_name>}, dont install packages individually at any cost
        >> note: example modify [ requirements.txt ] >> numpy==1.26.
        >> then [source venv_<project_name>/bin/activate] && [ pip install -r requirements ]
        >> dont stop untill you resolve all errors
        >> but dont install packages individually at any cost , do you understand? 
>> FRESH MEMORY 
    >> if image or its directory is shared, view each of the following image with FRESH eyes, before making any conclusion
    >> re-read ALL lines of current version of code, before making any modification 
>> no HARDCODE / FALLBACK
    >> do NOT HARDCODE or do NOT use FALLBACK [e.g - CPU, if OOM on GPU] mechanism at any cost
    >> [if its difficult to implement, user's requirement, explicitly say so]
    >> no CPU, only GPU
>> TEST
    >> create TEST file in [ unit_test/ ] ONLY existing directory 
    >> I will test it manually outside claude terminal, if execution_time > 2 mins
    >> note: for every modified code, run [py_compile, function calling, IMPORT calling, Redundancy, etc] tests >> View the results >> before making any claim about improvements
    > before building next code modules >> READ, analyze and explain/ EDA the output of previous code module


>> ORGANIZE / NAMING
    >> numbering "m"odules for sorting: [src/m01_<name>.py, src/m02_<name>.py, ] (start with letter "m" to not get into import error with number as prefix)
        >> for EACH python files in [ src/ ] directory
        >> for EACH of respective subfolder in [ outputs/ ] directory
        >> for EACH of respective tables in [ outputs/centralized.db ] database
        >> No TIMESTAMP in primary or secondary key, so that similar combination experiement, if re-run, should be replaced
    >> All log / data generated should be stored in [outputs/centralized.db] single datasource. No individual JSON or TEXT file for anyone of the scripts
    >> you may keep individual TABLE for each python script [src/m01_<name>.py, src/m02_<name>.py, ]
    >> so that next python files can use the output of previous python file directly from [outputs/centralized.db] single datasource
    >> avoid creating txt/json file for logs >> so we  wont have to do text matching  
    >> keep all python files in [ src/ ] directory

