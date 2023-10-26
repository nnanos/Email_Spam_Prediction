=======================================================================
Json Parser with FLEX and BISON tools
=======================================================================

Description
============

This repository contains an implementation of a Json Parser. 
Json files are well known and used in a variety of applications.
These files have a specific format which is described **here**.
The parser implemented here is accomplished with the help of some old but good tools **flex** and **bison**.
Json format can be described by the below BNF syntactic definition of the grammar of the language. ::


<programme> ::= <array> | <object>
<array> ::= “[]” | “[” <elements> “]”
<elements> ::= <value> | <value> “,” <elements>
<object> ::= “{}” | “{” <members> “}”
<members> ::= <pair> | <pair> “,” <members>
<pair> ::= STRING “:” <value>
<value> ::= STRING | NUMBER | <objetct> | <array> | “true” | “false” | “null”

 
 **Note**: STRING and NUMBER are tokens that meet certain specifications
 (regular expressions) and which have been produced by the lectical analyzer (flex). The same applies
 for the rest of the terminal symbols which have been enclosed in double quotation marks.


So the parser with one pass of the input checks: 


#. If the input is a syntactically correct json (satisfiying the above BNF grammar).

#. If the input meets the bellow requierements.

  * The “text” element should be up to 140 characters.
 
  * The “user” element should contain a unique “id” as a positive
    integer, and “name”, “screen_name”, “location” as alphanumeric.
 
  * The “created_at” element should be in the format “Day_name MMM DD
    XX:XX:XX YYYY” where Day_name = Monday, Tuesday, etc., M = Month,
    D = Day, XX:XX:XX the timestamp format and Y = Year. For the timestamp
    make sure that a reasonable result is obtained ( hours 0 to 23 , minutes and
    seconds from 0 to 60 ).
 
  * The “id_str” element should be alphanumeric and UNIQUE,
    however contain only one positive integer, e.g. “19487012”,
    "8623490". 



Usage
=============

	* SETUP
	
	Firstly to use the parser you have to execute the following commands. ::
	  
	  flex Scanner.l
	  bison -d -g -t json_parser.y
	  gcc -g lex.yy.c json_parser.tab.c -o parser
	
	
	These commands compile the flex and bison files and generates the parser executable.
	
	
	
	Equivalently you can execute the script compile_script.sh (which just contains the above commands). ::
	  
	  chmod +x compile_script.sh
	  ./compile_script.sh
	
	
	* EXAMPLES
	
	Every example that I present here can be reproduced because I provide all the files in this repository.
	
	 * I run the parser to check that a random json that I found `here <https://json.org/example.html`_ is a syntactically correct json file. ::
	 
	    ./parser "/Path/to/image/GOOD_json.txt"
	 
	   
	  The output of the parser is as expected the syntactically correct json file that used as the input and can be shown bellow. 
 

.. image: /Files_to_check_and_Images/json_structure_check/GOOD_JSON.png

   
  If I modify the GOOD_json.txt so that it doesnt have one double quote in line 3 (tittle field) then the output is:
  

.. image: /Files_to_check_and_Images/json_structure_check/BAD_JSON.png
 
 
    Now we see that the output is an error message and not just the input file as before.


	* Checking the **Text_field**


	* Checking the **created_at_field**


	* Checking the **id_str_field**


	* Checking the **user_field**



	I now present some examples of execution.




============

Free software: MIT license
============

