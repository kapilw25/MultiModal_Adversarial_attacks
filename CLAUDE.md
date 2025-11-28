>> PLAN MODE: 
    >> create a TODO list when working on complex tasks to track progress and remain on track
    >> do not be YES MAN >> do WEBSEARCH >> give me PROS and CONS of atleast 3  approaches 

>> INSTALLATION
    >> in virtual environment {venv_<project_name>}, dont install packages individually at any cost
        >> note: example modify [ requirements.txt ] >> numpy==1.26.
        >> then [source venv_<project_name>/bin/activate] && [ pip install -r requirements ]
        >> dont stop untill you resolve all errors
        >> but dont install packages individually at any cost , do you understand? 
>> no HARDCODE / FALLBACK
    >> np CPU, all heavy computaitons must be done on GPU
    >> do NOT HARDCODE or do NOT use FALLBACK [e.g - CPU, if OOM on GPU, if automation doest work, hardcode the answer] mechanism at any cost
    >> [if its difficult to implement, user's requirement, explicitly mention the limitations >> but no FALLBACK
>> TEST
    >> for every modified code, run [py_compile, Import check, AST based syntax check, function calling, IMPORT calling, Redundancy, etc] tests >> View the results >> before making any claim about improvements
    >> I will test it manually outside claude terminal, if execution_time > 2 mins
    >> before building next code modules >> READ, analyze and explain/ EDA the output of previous code module
    >> create TEST file in [ unit_test/ ] ONLY existing directory 


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
- do not add CLAUDE in git commit
- in notebook, I will do all the modificaitons MANUALLY

# Github
> git config user.name "kapilw25"
  git config user.email "kapilw25@gmail.com" 

#   Simple solution to overwrite everything:
```
git fetch && git reset --hard origin/main
```
What it does:
- git fetch - downloads remote changes
- git reset --hard origin/main - throws away ALL local changes and matches remote exactly

#   TensorBoard via SSH Tunnel
On Cloud Terminal:
```
tensorboard --logdir=tensorboard_logs --port=6006
```

On Local Terminal:
```
ssh -L 6006:localhost:6006 lambda_A100_40GB
```

Open Browser: http://localhost:6006

# NOTES
>> note: be brutally honest. You do not have to agree with me, unless I am correct. But do not LIE/ Halluciante too
>>  Keep explanation SHORT. I cant read verbose explanations
>> Remove all false advertising / STATIC prints


>> note: being Devil's advocate does NOT mean, hallucinate/fake-produce the mistakes which dont exist. If the code is correct , then accept it and move on
>> WEBSEARCH [not always] : if needed to find the universal practices in Ai/ML research world
- never modify @CLAUDE.md