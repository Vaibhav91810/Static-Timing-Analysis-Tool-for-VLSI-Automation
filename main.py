import numpy as np
import re
import argparse

import os


def write_output_file(inputs, outputs, gate_types, fanin, fanout, gates):
    with open('ckt_details.txt', 'w') as file:
        file.write(f"{len(inputs)} primary inputs\n")
        file.write(f"{len(outputs)} primary outputs\n")
        for gate_type, count in sorted(gate_types.items()):
            file.write(f"{count} {gate_type} gates\n")
        file.write("Fanout...\n")
        
        # Ensure outputs are in the correct format
        formatted_outputs = {f"OUTPUT-{o}": o for o in outputs}

        for gate, outs in sorted(fanout.items()):
            # Construct fanout labels, ensuring "OUTPUT-" prefix is included
            outs_labels = []
            for out in outs:
                if out in gates:
                    outs_labels.append(f"{gates[out]['type']}-{out}")
                elif f"OUTPUT-{out}" in formatted_outputs:
                    outs_labels.append(f"OUTPUT-{out}")

            # If a gate drives an output, append the output label
            for output_label in formatted_outputs:
                if output_label.endswith(gate):
                    outs_labels.append(output_label)
            
            gate_label = f"{gates[gate]['type']}-{gate}"
            file.write(f"{gate_label}: {', '.join(outs_labels)}\n")
        
        file.write("Fanin...\n")
        for gate, ins in sorted(fanin.items()):
            # Construct fanin labels
            ins_labels = [f"{gates[in_gate]['type']}-{in_gate}" if in_gate in gates else f"INPUT-{in_gate}" for in_gate in ins]
            gate_label = f"{gates[gate]['type']}-{gate}"
            file.write(f"{gate_label}: {', '.join(ins_labels)}\n")


def parse_bench_file(file_path):
    inputs = []
    outputs = []
    gates = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            elif line.startswith('INPUT'):
                input_name = line.split('(')[1].strip(')')
                inputs.append(input_name)
            elif line.startswith('OUTPUT'):
                output_name = line.split('(')[1].strip(')')
                outputs.append(output_name)
            else:
                parts = line.split('=')
                gate_name = parts[0].strip()
                gate_type, connections = parts[1].strip().split('(')
                connections = [conn.strip() for conn in connections.strip(')').split(',')]
                gates[gate_name] = {
                    'type': gate_type,
                    'connections': connections
                }

    return inputs, outputs, gates

def analyze_circuit(inputs, outputs, gates):
    gate_types = {}
    fanin = {gate: [] for gate in gates}
    fanout = {gate: [] for gate in gates}

    for gate_name, gate_info in gates.items():
        gate_type = gate_info['type']
        gate_types[gate_type] = gate_types.get(gate_type, 0) + 1

        for conn in gate_info['connections']:
            if conn in gates: 
                fanout[conn].append(gate_name)
            fanin[gate_name].append(conn)

    for output in outputs:
        for gate_name, gate_info in gates.items():
            if output in gate_info['connections']:
                fanout[gate_name].append(output)

    return gate_types, fanin, fanout


def parse_liberty_file(liberty_file_path, mode):
    with open(liberty_file_path, 'r') as file:
        content = file.read()
    
    content = re.sub(r'\\\n', '', content)

    cell_blocks = re.findall(r'cell\s*\(([^)]+)\)\s*{([^}]+)}', content, re.DOTALL)

    parsed_data = {}
    for cell_name, cell_body in cell_blocks:
        cell_name = cell_name.strip()
        parsed_data[cell_name] = {
            'index_1': [],
            'index_2': [],
            'values': []
        }

        index_1_match = re.search(r'index_1\s*:\s*"([^"]+)"', cell_body)
        index_2_match = re.search(r'index_2\s*:\s*"([^"]+)"', cell_body)
        values_match = re.search(r'{}.*?values\s*:\s*"([^"]+)"'.format('cell_delay' if mode == 'delays' else 'output_slew'), cell_body, re.DOTALL)
        
        if index_1_match:
            parsed_data[cell_name]['index_1'] = index_1_match.group(1).split(',')
        if index_2_match:
            parsed_data[cell_name]['index_2'] = index_2_match.group(1).split(',')
        if values_match:
            parsed_data[cell_name]['values'] = [item.strip() for sublist in [v.split(',') for v in values_match.group(1).split(';')] for item in sublist]

    return parsed_data

def write_to_file(data, mode, filename):
    with open(filename, 'w') as file:
        for cell_name, cell_data in data.items():
            file.write(f'cell: {cell_name}\n')
            file.write(f'input slews: {", ".join(cell_data["index_1"])}\n')
            file.write(f'load cap: {", ".join(cell_data["index_2"])}\n')
            file.write(f'{mode}:\n')
            for value in cell_data['values']:
                file.write(f"{value}\n")
            file.write('\n')


def parse_liberty_file_HA(liberty_file_path):
    with open(liberty_file_path, 'r') as file:
        lines = file.readlines()

    full_text = ""
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.endswith("\\"):
            full_text += stripped_line[:-1].strip() + " "
        else:
            full_text += stripped_line + "\n"

    pattern = r'cell \((.*?)\) \{\s*capacitance\s*:\s*([\d.]+);\s*cell_delay\(Timing_7_7\) \{\s*index_1\s*\(("(?:[\d.,\s]+)"(?:,\s*"[^"]+")*)\);\s*index_2\s*\(("(?:[\d.,\s]+)"(?:,\s*"[^"]+")*)\);\s*values\s*\(\s*("(?:[\d.,\s]+)"(?:,\s*"[^"]+")*)\s*\);\s*\}\s*output_slew\(Timing_7_7\) \{\s*index_1\s*\(("(?:[\d.,\s]+)"(?:,\s*"[^"]+")*)\);\s*index_2\s*\(("(?:[\d.,\s]+)"(?:,\s*"[^"]+")*)\);\s*values\s*\(\s*("(?:[\d.,\s]+)"(?:,\s*"[^"]+")*)\s*\);\s*\}\s*\}\s*'
    pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
    matches = pattern.findall(full_text)
    data = []

    d = {}
    for match in matches:
        cell_name, capacitance, index_1, index_2, values_delay, index_1_slew, index_2_slew, values_slew = match
        string_delay = values_delay.strip('"').split(",")
        string_delay = [value.replace('"', '').replace(' ','') for value in string_delay]
        string_delay = [float(value) for value in string_delay]
        string_slew = values_slew.strip('"').split(",")
        string_slew = [value.replace('"', '').replace(' ','') for value in string_slew]
        string_slew = [float(value) for value in string_slew]

        values_array_delay = np.array(string_delay, dtype=float)
        values_array_slew = np.array(string_slew, dtype=float)

        grid_7x7_delay = values_array_delay.reshape(7, 7)
        grid_7x7_slew = values_array_slew.reshape(7, 7)

        d[cell_name] = {
            "capacitance": float(capacitance),
            "delays": {
                "index_1": index_1.strip('"').split(","),
                "index_2": index_2.strip('"').split(","),
                "values": grid_7x7_delay
            },
            "slews": {
                "index_1": index_1_slew.strip('"').split(","),
                "index_2": index_2_slew.strip('"').split(","),
                "values": grid_7x7_slew
            }
        }

    return d


def interpolation(data, gate, mode, input_slew, load):  #input_slew=index1, load=index2
    r1 = 0
    r2 = 0
    c1 = 0
    c2 = 0

    for i in range(1, len(data[gate][mode]['index_1'])):
        if input_slew >= float(data[gate][mode]['index_1'][i-1]) and input_slew < float(data[gate][mode]['index_1'][i]):
            r1 = i-1
            r2 = i
            break
    
    for i in range(1, len(data[gate][mode]['index_2'])):
        if load >= float(data[gate][mode]['index_2'][i-1]) and load < float(data[gate][mode]['index_2'][i]):
            c1 = i-1
            c2 = i
            break
    
    v11 = float(data[gate][mode]['values'][r1][c1])
    v12 = float(data[gate][mode]['values'][r1][c2])
    v21 = float(data[gate][mode]['values'][r2][c1])
    v22 = float(data[gate][mode]['values'][r2][c2])
    T1 = float(data[gate][mode]['index_1'][r1])
    T2 = float(data[gate][mode]['index_1'][r2])
    C1 = float(data[gate][mode]['index_2'][c1])
    C2 = float(data[gate][mode]['index_2'][c2])

    record = {'v11': v11, 'v12': v12, 'v21': v21, 'v22': v22, 'T1': T1, 'T2': T2, 'C1': C1, 'C2': C2}

    num = v11*(C2 - load)*(T2 - input_slew) + v12*(load - C1)*(T2 - input_slew) + v21*(C2 - load)*(input_slew - T1) + v22*(load - C1)*(input_slew - T1)
    deno = (C2 - C1)*(T2 - T1)
    return num/deno



from collections import defaultdict, deque

def read_edges_from_file(file_path):
    edges = []
    processing_fanin = False  # State variable to track if we're processing fanin or fanout

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Check for the start of fanin or fanout sections
            if line.startswith("Fanout"):
                processing_fanin = False
                continue
            elif line.startswith("Fanin"):
                processing_fanin = True
                continue
            
            # Skip empty lines or non-relevant metadata
            if not line or ':' not in line:
                continue

            parts = line.split(":")
            if len(parts) == 2:
                node = parts[0].strip()
                connections = [conn.strip() for conn in parts[1].split(",")]

                # Depending on section, add edges in reverse for fanin
                for conn in connections:
                    if processing_fanin:
                        edges.append((conn, node))
                    else:
                        edges.append((node, conn))

    return edges



from collections import defaultdict, deque

def perform_topological_sort(edges):
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    # Create the graph and calculate in-degrees
    for src, dest in edges:
        graph[src].append(dest)
        in_degree[dest] += 1
        in_degree[src] += 0  # Ensure all source nodes are in the in_degree map

    # Find the start nodes (nodes with zero in-degree)
    start_nodes = [node for node in in_degree if in_degree[node] == 0]
    sorted_nodes = []
    queue = deque(start_nodes)

    # Process the graph
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        
        for adjacent in graph[node]:
            in_degree[adjacent] -= 1
            if in_degree[adjacent] == 0:
                queue.append(adjacent)

    # Check for a cycle (if not all nodes were processed)
    if len(sorted_nodes) != len(in_degree):
        return "The graph contains a cycle, and a topological sort is not possible."
    else:
        return sorted_nodes


def parse_inputs_outputs(file_path):
    inputs = []
    outputs = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        found_fanin = False
        found_fanout = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("Fanin"):
                found_fanin = True
                found_fanout = False
                continue
            elif line.startswith("Fanout"):
                found_fanin = False
                found_fanout = True
                continue
            
            if found_fanin:
                parts = line.split(":")
                if len(parts) == 2:
                    inputs.extend([inp.strip() for inp in parts[1].split(",")])
            elif found_fanout:
                parts = line.split(":")
                if len(parts) == 2:
                    outputs.extend([out.strip() for out in parts[1].split(",")])
    
    final_input = set()
    final_output = set()
    for i in inputs:
        if i.startswith('INPUT'):
            final_input.add(i)
    
    for i in outputs:
        if i.startswith('OUTPUT'):
            final_output.add(i)

    return list(final_input), list(final_output)
    

def calculate_index2_from_text(data, file_path, fanout, gates):
    index2 = {}
    gate_cap = {"NAND": 1.599032, "NOR": 1.714471, "AND": 0.918145, "OR": 0.946814, "XOR": 2.411453, "INV": 1.700230, "BUF": 0.974659, "NOT": 1.700230, "BUFF": 0.974659}

    for key in fanout.keys():
        key_val = 0
        if len(fanout[key]) > 0:
            for k in fanout[key]:
                key_val += gate_cap[gates[k]['type']]
        else:
            key_val = 4*gate_cap['INV']
        name = gates[key]['type'] + '-' + key
        index2[name] = key_val
    # print("index2 ",index2)

    return index2


dictionary = {}

def calculate_index1_from_text(file_path,index2,parsed_data, fanin, gates, inputs, sorted_order):
    index1 = {}
    gate_types = {"NAND": "NAND2_X1", "NOR": "NOR2_X1", "AND": "AND2_X1", "OR": "OR2_X1", "XOR": "XOR2_X1", "INV": "INV_X1", "BUF": "BUF_X1", "NOT": "INV_X1", "BUFF": "BUF_X1"}
    for gate in sorted_order:
        g_name = gate.split('-')[0]
        if g_name == 'INPUT' or g_name == 'OUTPUT':
            continue
        num = gate.split('-')[1]
        temp = []
        for conn in fanin[num]:
            if conn in inputs:
                temp.append(0.002)
            else:
                g = gates[conn]['type'] + '-' + conn
                temp.append(dictionary[g]["op_slew"])
                
        cell_delay = interpolation(parsed_data, gate_types[g_name], 'delays', max(temp), index2[gate])
        op_slew = interpolation(parsed_data, gate_types[g_name], 'slews', max(temp), index2[gate])
        dictionary[gate] = {"cell_delay":cell_delay*1000, "op_slew":op_slew}

    # print("dictionary ", dictionary)

    return index1


def main():

    parser = argparse.ArgumentParser(description='Perform STA on a given circuit and liberty file.')
    parser.add_argument('--read_ckt', type=str, help='Path to the .bench file to parse')
    parser.add_argument('--read_nldm', type=str, help='The liberty file to read from')
    args = parser.parse_args()

    if args.read_ckt and args.read_nldm:
        inputs, outputs, gates = parse_bench_file(args.read_ckt)
        gate_types, fanin, fanout = analyze_circuit(inputs, outputs, gates)
        write_output_file(inputs, outputs, gate_types, fanin, fanout, gates)
        while not os.path.exists("./ckt_details.txt"):
            pass

        file_path = 'ckt_details.txt'
        edges = read_edges_from_file(file_path)
        sorted_order = perform_topological_sort(edges)
        parsed_data = parse_liberty_file_HA(args.read_nldm)
        index2 = calculate_index2_from_text(parsed_data, file_path, fanout, gates)

        index1 = calculate_index1_from_text(file_path,index2,parsed_data, fanin, gates, inputs, sorted_order)

        def parse_gate_connections(file_path):
            gate_inputs = {}
            outputs_to_gates = {}

            with open(file_path, 'r') as file:
                lines = file.readlines()

            current_section = None
            for line in lines:
                line = line.strip()
                if 'Fanout...' in line:
                    current_section = 'fanout'
                    continue
                elif 'Fanin...' in line:
                    current_section = 'fanin'
                    continue
                elif not line or ':' not in line:
                    continue

                parts = line.split(': ')
                if len(parts) != 2:
                    continue

                gate, connected_gates_str = parts
                connected_gates = connected_gates_str.split(', ')

                if current_section == 'fanin':
                    gate_inputs[gate] = connected_gates
                elif current_section == 'fanout':
                    outputs_to_gates[gate] = connected_gates

            for gate, outputs in outputs_to_gates.items():
                for output_gate in outputs:
                    if output_gate not in gate_inputs:
                        gate_inputs[output_gate] = []
                    if gate not in gate_inputs[output_gate]:
                        gate_inputs[output_gate].append(gate)

            all_gates = set(gate_inputs.keys()) | set(outputs_to_gates.keys())
            for gate in all_gates:
                if gate not in gate_inputs:
                    gate_inputs[gate] = []

            sorted_gate_inputs = {gate: gate_inputs[gate] for gate in sorted(gate_inputs)}

            return sorted_gate_inputs

        gate_inputs = parse_gate_connections(file_path)

        # Display the connections for each gate, ensuring outputs are included
        # for gate, inputs in gate_inputs.items():
        #     print(f'{gate}: {inputs}')


        def read_circuit_description(file_path):
            graph = defaultdict(list)
            processing_fanin = False

            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    if line == "Fanin...":
                        processing_fanin = True
                        continue
                    if line == "Fanout...":
                        processing_fanin = False
                        continue

                    parts = line.split(': ')
                    if len(parts) != 2:
                        continue
                    key, values_part = parts
                    values = values_part.split(', ')
                    if processing_fanin:
                        for value in values:
                            graph[value].append(key)
                    else:
                        for value in values:
                            graph[key].append(value)

            return graph

        def find_paths(graph, start, end, path=[]):
            path = path + [start]
            if start == end:
                return [path]
            if start not in graph:
                return []
            paths = []
            for node in graph[start]:
                if node not in path:
                    newpaths = find_paths(graph, node, end, path)
                    for newpath in newpaths:
                        paths.append(newpath)
            return paths

        def generate_paths_from_file(file_path):
            graph = read_circuit_description(file_path)
            input_nodes, output_nodes = parse_inputs_outputs(file_path)
            all_paths_set = set()
            for inp in input_nodes:
                for outp in output_nodes:
                    for path in find_paths(graph, inp, outp):
                        all_paths_set.add(tuple(path))

            all_paths = [list(path) for path in all_paths_set]
            return all_paths

        paths = generate_paths_from_file(file_path)
        paths= sorted(paths)
        
        # for path in paths:
        #     print(" -> ".join(path))
        
        arrival_times = {}
        circuit_delay = 0
        for path in paths:
            for count, ele in enumerate(path):
                if ele in arrival_times and ele.startswith("INPUT"):
                    continue
                elif ele not in arrival_times and ele.startswith("INPUT"):
                    arrival_times[ele] = 0
                elif ele.startswith("OUTPUT"):
                    arrival_times[ele] = arrival_times[path[count-1]]
                    circuit_delay = max(circuit_delay, arrival_times[ele])
                else:
                    if ele not in arrival_times:
                        arrival_times[ele] = 0
                    arrival_times[ele] = max(arrival_times[ele], dictionary[ele]['cell_delay'] + arrival_times[path[count - 1]])

        rt = circuit_delay*(1.1)

        backtraversal = {}
        slack_track = {}
        for path in paths:
            for count, ele in enumerate(path[::-1]):
                if ele.startswith("OUTPUT"):
                    backtraversal[ele] = rt
                    slack_track[ele] = rt
                else:
                    if ele not in backtraversal:
                        backtraversal[ele] = float('inf')
                        slack_track[ele] = float('inf')
                    cell_delay = 0 if ele.startswith("INPUT") else dictionary[ele]['cell_delay']
                    backtraversal[ele] = min(backtraversal[ele], backtraversal[path[::-1][count - 1]] - cell_delay)
                    slack_track[ele] = min(slack_track[ele], backtraversal[path[::-1][count - 1]])
        
        final_slack = {}
        min_slack = {"value" : float('inf')}
        for key in backtraversal.keys():
            final_slack[key] = round(slack_track[key] - arrival_times[key], 14)
            if key.startswith("OUTPUT") and final_slack[key] < min_slack["value"]:
                min_slack["value"] = final_slack[key]
        
        final_path = []     
        for path in paths:
            rp = path[::-1]
            for i in rp:
                if final_slack[i] != min_slack["value"]:
                    break
                if i.startswith("INPUT") and final_slack[i] == min_slack["value"]:
                    final_path.append(path)

    with open('ckt_traversal.txt', 'w') as f:
        f.write(f"Circuit delay: {circuit_delay} ps\n\n")
        f.write("Gate slacks:\n")
        for k in final_slack.keys():
            f.write(f"{k}: {final_slack[k]} ps\n")
                
        f.write("\nCritical path:\n")
        f.write(", ".join(final_path[0]))

    # print("Delays ", dictionary)
    # print("arrival_times", arrival_times)
    # print("slack_track ", slack_track)
    # print("FINAL SLACK ", final_slack)
    print(final_path[0])

if __name__ == '__main__':
    main()