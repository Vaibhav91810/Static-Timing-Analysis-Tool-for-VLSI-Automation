Installation Requires:
-Python 3.7 or later
-argparse module (usually included in standard Python installations)

Run the script from the command line with the following options:

To analyze a circuit description:
python3.7 parser.py --read_ckt <path_to_bench_file>

To extract delay data from a Liberty file:
python3.7 parser.py --delays --read_nldm <path_to_liberty_file>

To extract slew data from a Liberty file:
python3.7 parser.py --slews --read_nldm <path_to_liberty_file>

To print the critical path and perform STA on the ckt use command “python3.7 main.py –-read_ckt c17.bench –-read_nldm 
sample_NLDM.lib” 
which will produce a file called “ckt_traversal.txt”,

Output:
- When analyzing a circuit description, the script generates a file named "ckt_details.txt" containing circuit information (inputs, outputs, gate types, fanin, and fanout).
- When extracting delay or slew data, the script generates a file named "delays_LUT.txt" or "slews_LUT.txt", respectively, containing the extracted data.


