###### tags: `Linux`


Bash
===

# Bash Script
### Pipes and redirection
- pipes send the result of one process to another ( send the output of one commend to the next command instead of displaying on the screen)
    - ```|```: pipe
    - ```> >> < <<```: redirection
```bash=
# wc: word count
ls | wc -l # ls the number of the folder or file in the current directory

# redirection 將結果輸出到list.txt
ls > list.txt
# add the information onto the end of the existing file
ls >> list.txt
# redirect the standard output with one  & standard error with 2 輸出 
# std output: 1, std err: 2
ls /notreal 1>output.txt 2>error.txt
# take information from a file
cat < list.txt

# use the command version of ```echo``` instead of the builtin version, we would write
command echo
```


![](https://i.imgur.com/CiK3QG6.png =300x)
```echo```: print text to the standard output
```~```: represents the user's $HOME environment variable
```{}```: create sets or ranges
```touch <new file>```: create a new file

```bash=
# 不能有空格
echo /tmp/{1,2,4}/file.txt 
# Output: /tmp/1/file.txt /tmp/2/file.txt /tmp/4/file.txt
echo /tmp/{1..4}/file.txt
# Output: /tmp/1/file.txt /tmp/2/file.txt /tmp/3/file.txt /tmp/4/file.txt

echo {00..100}
echo {00..30..3}
# Output: 00 03 06 09 12 15 18 21 24 27 30
echo {a..z..2}
# Output: a c e g i k m o q s u w y
echo {cat,dog}_{01..05}
# Output: cat_01 cat_02 cat_03 cat_04 cat_05 dog_01 dog_02 dog_03 dog_04 dog_05
```

```${...}```: put the output of one command inside another **(command substitution)**
```bash=
# retrieve and transform stored values
a="HelloWorld"
echo $a
# Output: HelloWorld
b=BashCommand
echo ${b}
# Output: BashCommand 
char_list="abcdefghijk"
echo ${greeting:6:3} # start print at the 6th character and 3 character
man bash # check the paramter using "/" and input the keyword you want to search

greeting="hello there"
echo ${greeting/there/everybody} # replace there with everybody 
# Output: hello everybody
echo ${greeting//e/_} # replace all using "//"
# Output: h_llo th_r_
echo ${greeting/e/_} # replace all using "//"
# Output: h_llo there
```

```$(...)```: puts the output of one command inside another. Often used together with string manipulation tools. e.g. path, file size, IP address

```bash=
uname -r # get the release version of the kernel
echo "The kernel is $(uname -r)."
# Output: The kernel is 3.0.4(0.338/5/3).
echo "Result: $(python3 -c 'print("Hello from Python!")' | tr [a-z] # transform
```

```$((..))```: **arithmetric expansion** does math
```bash=
echo $((2 + 2))
# Output: 4
```


## Two ways to run bash script (programming with Bash)
Bash script run inside of a noninteractive shell
1. **Bash script**
    - text file that contains a series of commands
    - ```bash script.sh```
2. **Executable bash script (general way)**
    - includes a **sheband** as the first line **```#!/usr/bin/env bash```** (tells the shell when the script is run & the path to whatever program should run the script)
    - make executable with **```chmod +x myscript```**
    - run locally with ./myscript or myscript if in the $PATH

- keep in mind of ```declare``` for remind you some situation e.g. read only
```./script.sh```: run the script on the right directory
```declare -p```: show all the variables that have been set in the current session

```bash=
# script.sh
#!/usr/bin/env bash
myvar="Hello"
echo "The value of the myvar variable is: $myvar"
myvar="World"
echo "The value of the myvar variable is: $myvar"

declare -r myname="Eric" # read only (cannot change the variable)
echo "The value of the myname variable is: $myname"
myname="ASD"
echo "The value of the myname variable is: $myname"

declare -l lowerstring="This is some TEXT" # transform into lower case
echo "Lower string is: $lowerstring"
lowerstring="Let's have the LOWER CASE"
echo "Lower string is: $lowerstring"
 
declare -u upperstring="This is some TEXT" # transform into upper case
echo "Upper string is: $upperstring"
upperstring="Let's have the UPPER CASE"
echo "Upper string is: $upperstring"

```
```=+
# Output
The value of the myvar variable is: Hello
The value of the myvar variable is: World
The value of the myname variable is: Eric
./myscript.sh: line 14: myname: readonly variable
The value of the myname variable is: Eric
Lower string is: this is some text
Lower string is: let's have the lower case
Upper string is: THIS IS SOME TEXT
Upper string is: LET'S HAVE THE UPPER CASE
```

### number operation
Supported Arithmetric Operations ```+, -, *, /, %, **```

**```$((...))```**: **(arithmetic expansion)** return the result of mathematical operations
**```((...))```**:  **(arithmetic evaluation)** perform calculation and change the value of variables

Bash can do integer math, but not decimal or fractional
To do more precise math, consider using ```bc``` or ```awk```
```bash=
a=3
((a+=3))
echo $a
# Output: 6
((a++))
echo $a
# Output: 7
((a--))
echo $a
# Output: 6
a=$a+2 # bash treat this as a string not a number
# Output: 6+2

# To prevent bash treating the number as string, we use declare
declare -i b=3 # treat as integer
b=$b+3 # no black here
# Output: 6

# for more precise math
declare -i c=1
e=$(echo "scale=3; $c/$b" | bc) # scale: 位數
echo $e
# Output: 0.166

echo $RANDOM
# to get reduce range from 1 to 20
echo $((1 + RANDOM % 20)) # range from 1 to 20
```

### test value (regular test vs. extended test)
**```[ ... ]```**: an alias for test and is used to **test, evaluate, expression** (builtin function) **注意前後要空格**
use ```help test``` for more information
- 0: success
- 1 or other number: failure

**```$?```**: read the value of the return status

```bash=
# test whether a file exist
if test -f "$FILE"; then
    echo "$FILE exists."
fi
# another way
if [ -f "$FILE" ]; then
    echo "$FILE exists."
fi
# test string operator (=, !=, >, <)
# >: str1 sorts before str2 lexicographically
[ "cat" = "dog" ]; echo $?
[ "cat" = "cat" ]; echo $?

[ 4 -lt 5 ]; echo $? # less than
# Output: 0
[ 5 -ge 3]; echo $? # greater or equal to
# Output: 0
[ ! 5 -ge 3]; echo $? # not greater or equal to
# Output: 1
```

**```[[ ... ]]```**: extended test
- same operations as test, and add a few other features 
- can create a more complex logic (multiple conditions is allowed in one single test)

```bash=
# whether my home directory is directory & whether the Bash binary exists
[[ -d ~ && -a /bin/bash ]]; echo $?
# Output: 0
[[ -d ~ && -a /bin/mash ]]; echo $?
# Output: 1
[[ -d ~ || -a /bin/bash ]]; echo $?
# Output: 0
[[ -d ~ ]] && echo ~ is a directory
# Output: /home/mobaxterm is a directory
[[ -d /bin/bash ]] && echo ~ is a directory
# since /bin/bash is not a directory, so doesn't output anything

# regular expression
[[ "cat" =~ c.* ]]; echo $?
# Output: 0
```

#### Formatting and text styling output
**```echo -e```**: control special character

change color: e.g. ```\033[5mMyStIcAl sPhErE\033[0m```
```\t```: tab, ```\n```: new line
```bash=
echo -e "Name\t\tNumber"; echo -e "Eric\t\t123"
# Output
#Name            Number
#Eric            123

echo -e "This text\nbreaks over\nthree lines"
# Output
#This text
#breaks over
#three lines
```

**```printf "..."```**: Output text using placeholders and formatting
```bash=
printf "The results are: %d and %d\n" $((2+2)) $((4/1))
```
![](https://i.imgur.com/nOWmYCR.png =600x)

#### Array in bash
- support indexed and associative arrays
- remember only one-dimension array is supported in bash (no nested array)
```bash=
# declare indexed array (use ```-a```)
declare -a snacks=("apple" "banna" "orange") 
echo ${snacks[2]}
# Output: orange
snacks+=("mango")
echo ${snacks[@]} # @ output the whole array

snacks[6]="grapes"

for i in {0..6}; do echo "$i: ${snacks[$i]}"; done
# Output
#0: apple
#1: banna
#2: orange
#3: mango
#4:
#5:
#6: grapes

# declare associative array  (use ```-A```)
declare -A office
office[city]="Hsinchu"
office["building name"]="ABC"
echo ${office["building name"]} is in ${office[city]}
```

### Make a script that generate a system report
- use some standard tools
    - ```df```: find storage utilization
    - ```free```: find out the memory
    - use ```awk``` or ```sed``` to extract text from output

```bash=
#!/usr/bin/env bash
# Briefly summarize of system information

freespace=$(df -h / | awk 'NR==2 {print $4}')
freememory=$(free -h | awk 'NR==2 {print $4}')

printf "Current system report is summarized below\n"
# printf -v logdate "%(%Y-%m-%d)T"
printf "\tKernel Release:\t%s\n" $(uname -r)
printf "\tBash Version:\t%s\n" $BASH_VERSION
printf "\tFree Storage:\t%s\n" $freespace
printf "\tFree Memory:\t%s\n" $freememory
```

```=+
Output
Current system report is summarized below
./system_report.sh: line 7: /home/mobaxterm: is a directory
        Kernel Release: 3.0.4(0.338/5/3)
        Bash Version:   4.1.17(0)-release
        Free Storage:
        Free Memory:
```

## Bash Control Structure
### Condition statements
```bash=
if ...
then
    ...
fi
# if ... else if ... else 
if ...; then
    ...
elif ...; then
    ...
else
    ...
fi
```
![](https://i.imgur.com/KmmIooW.png)
### Loops (while, until, for)
- while
```bash=
echo "While Loop"
declare -i n=0
while (( n<10 ))
do
        echo "n: $n"
        (( n++ ))
done
```
- until
```bash
echo "Until Loop"
declare -i m=0
until (( m==3 ))
do
        echo "m: $m"
        (( m++ ))
done
```
- for
```bash=
echo "For loop"
for i in {0..6}
do
    echo $i
done


declare -a fruits=("apple" "banana" "cherry")
for i in ${fruits[@]}
do
        echo $i
done
```
### Case statement

```bash=
animal="dog"
case $animal in 
    cat) echo "Feline";;
    dog|puppy) echo "Canine";;
    *) echo "No match"   # else
esac # case spell backward
```
```=+
Output:
Canine
```
### Function
```bash=
#!/usr/bin/env bash
greet() {
    echo "Hi there, $1. What a nice $2"
}

# argument start from $1
greet Eric morning
greet Andy evening
```

- $@: represent the list of arguments given to a function
- $FUNCNAME: represents the name of the function

```bash=
display_result(){
    declare -i i=1
    for fruit in $@; do
        echo "$i: ${fruit}"
        (( i+=1 ))
    done
}

display_result apple banana cherry grapes
```

### File I/O
- common usage for log file
- Write to files with output redirection operators (```>``` and ```>>```)
    - ```echo "abc" > output.txt```: overwrites the contents
    - ```echo "abc" >> output.txt```: append to the end of txt
- Read from files with input redirection (```<```) and read command
    - ```while read line; do echo $line; done < input.txt```
    - **can read log files with regular expression**
```bash=
# f represents line
while read f 
    do echo " $f"
done < ~textfile.txt
```
### Compose a script that control structures to show random replies
- include quote viewer, a dice roll, or a card draw




## Interact with the user

### Arguments
- text that represent a string, filename, and so on
- are represented by numbered variables ($1, $2, and so on)


```bash=
#!/usr/bin/env bash
# $0: represent the bash script
echo "The $0 script got the argument $1"
echo "Argument 2 is $2"
```
```bash=
./myscript Banana Cherry
```

- loop for all arguments
```bash=
for i in $@
do
    echo $i
done
```
### Options
- allow us to pass information into a script from the CLI
- Are a combination of a dash and a letter (like ```-u``` or ```-p```)
- are **accessed using the getopts keyword**
- can accept arguments of their own
- can be **specified and used in any order**

```bash=
#!/usr/bin/env bash
while getopts u:p: option; dp
    case $option in
        u) user=$OPTARG;;
        p) pass=$OPTARG;;
    esac

echo "user: $user / pass: $pass"
``` 
```bash=+
./myscript -u Eric -p password123
```

- used to enable or disable certain features of a script
```bash=
# ab (flags without colons after them that means I just wanna know whether these flags are used)
while getopts u:p:ab option; dp
    case $option in
        u) user=$OPTARG;;
        p) pass=$OPTARG;;
        a) echo "got the A flag";;
        b) echo "got the B flag";;
    esac

echo "user: $user / pass: $pass"
```
```bash=
./myscript -u Eric -p password123 -a
./myscript -u Eric -p password123 -ab
```
- add ```:``` in front of ```u:p:```
```bash=
while getopts :u:p:ab option; dp
    case $option in
        u) user=$OPTARG;;
        p) pass=$OPTARG;;
        a) echo "got the A flag";;
        b) echo "got the B flag";;
        ?) echo "I don't know what $OPTARG is"
    esac

echo "user: $user / pass: $pass"
```
### Get inpuut during execution
- script often need input as they run 
- the read keyword allows us to gather input, pausing the script unit input is provided
- input is stored in a variable
- allows user input
```bash=
echo "What is your name?"
read name
echo "What is your password?"
read -s pass  # secret (not show on the screen)
read -p "What's your favorite animal? " animal

echo "name: $name, pass: $pass, animal: $animal"
help read
```

- ```select```

```bash=
echo "Which animal"
select animal in "cat" "dog" "bird" "fish"
do 
    echo "You selected $animal"
    break
done
```

### Ensure a response
- What if the user will simply ignore our request. e.g. just by pressing ```Enter``` ignoring input prompts. Might cause problem with empty information

- Sol1: give it a default value to prevent empty situation

```bash=
#!/usr/bin/env bash
read -ep "Favorite color? " -i "Blue" favcolor
echo $favcolor
```

- Sol2: use condition to provide users the information
```bash=
#!/usr/bin/env bash
# $#: the number of arguments provided at the command line
if (($# < 3)); then
    echo "This command requires 3 arguments:"
    echo "username, userid, and favorite number."
else
    echo "username: $1"
    echo "userid: $2"
    echo "favorite number: $3"
fi
```

- Sol3: use loops to continue without specifying some kind of input
    - It might be irritated for the users, so recommend to give it a default value
```bash=
read -p "Favorite animal?" fav
while [[ -z $fav ]] # check whether it is empty -z
do
    read -p "Cannot be empty! " fav
done
echo "$fav was selected."
```

```bash=
# give defualt value cat    [...]: default format
read -p "Favorite animal? [cat]" fav
while [[ -z $fav ]] # check whether it is empty -z
do
    fav="cat" # set the value equal to whether the default would be
done
echo "$fav was selected."
```

- ```=~```: regular expression 
```bash=
# {4}: 4 digits, ask 
read -p "What year? [nnnn] " year
until [[ $year =~ [0-9]{4} ]]; do
    read -p "A year, please! [nnnn] " year
done
echo "Selected year: $year"
```

## Bash in the real application
### Trouble Shooting
- tips: read the errors carefully, observe line numbers in errors
- check quotes and escaping
    - single quotes ('') and double quotes ("")
- check spacing in tests
    - ```[[$a-gt3]]``` will fail but [[ $a -gt 3 ]] will work
- check closure of expansions and substitutions
- add ```echo``` statement to keep track of parameter flow
- use the ```true``` and ```false``` built-ins to troubleshoot logic (to guarantee a successful or failed exit code)
- break down complex scripts into smaller parts
- Tell Bash to print out commands before it executes them
        - ```set -x``` and turn it off, you can use ```set +x```

### Ensuring script portability (compatibility)
- check the user's Bash version before running the script
- ```$BASH_VERSION```
```bash=
[[ ! $BASH_VERSION -ge 4 ]] && echo "You'll need to update to Bash 4+." && exit
```
- check if the user has nonstandard tools your script uses
    - ```[[ ! -a $(which nmap) ]] && echo "This script use Nmap, which was not found on this system." && exit```
- write the script so that it's compatible with the Bourne Shell (sh) instead of just Bash

[SHELL SCRIPT](https://fsl.fmrib.ox.ac.uk/fslcourse/lectures/scripting/all.htm)

[start here](https://www.linkedin.com/learning/learning-bash-scripting-2/ensuring-a-response?autoSkip=true&autoplay=true&resume=false)



### check the permission

