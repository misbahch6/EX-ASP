import argparse
from unittest import result
import numpy as np
import pandas as pd
from clingo import Control
from clingo.symbol import Number
from clingo.ast import parse_files, parse_string, ProgramBuilder
from clingolpx import ClingoLPXTheory
import time
from statistics import mean 
import os

# Constants
DEBUG = True
TIMEOUT = 60
target_variable = 'target'
whichclass = 1

def asp_inputs(prg,dom_of_each_var,input_var):
    # Create input var and set their domains
    for i, var in enumerate(input_var):
        prg += f'var({var}).\n'
        for dom in dom_of_each_var[i]:
            prg += f'dom({var},"{dom}").\n'
        prg += '\n'
    prg += '\n'
    return prg

def asp_hidden(prg,weights,biases,layer_index,size,input_var):
    # Create binary, real variables and formulas

    for i in range(size):
        node = 'y' if size == 1 else f'h{layer_index}{i}'
        prg += f'relu({node},"{biases[i]}").\n'
    prg += '\n'

    for i, row in enumerate(weights):
        for j, value in enumerate(row):
            if len(weights) == 1:
                prg += f'elem(y,"{value}",h{layer_index-1}{j}).\n'
            else:
                prg += f'elem(h{layer_index}{i},"{value}",{input_var[j]}).\n'
        prg += '\n'
    return prg


def nn_asp_explain(prg,o,W,V,finish_time):

    # Fix input values
    prg += '\n%Fix input values\n'
    for f in W:
        prg += f'input({f},"{V[f]}").\n'
    
    # output explanation 
    prg += '\n%Output explanation\n'
    prg += 'output(y,lt,"1").\n' if o else 'output(y,gt,"0").\n'

    #print(prg)
    if DEBUG:
        assign = ' '.join(f'{f}={str(V[f])}' for f in W)
        output = 'o=1' if o else 'o=0'
        print(f'calling check with: {assign} and {output}')

    thy = ClingoLPXTheory()
    thy.configure('strict', 'on')
    ctl = Control()#['--output-debug=text'])
    thy.register(ctl)
    
    with ProgramBuilder(ctl) as bld:
        parse_files(["encodings.lp"], lambda ast: thy.rewrite_ast(ast, bld.add))
        parse_string(prg, lambda ast: thy.rewrite_ast(ast, bld.add))

    def on_model(model):
        thy.on_model(model)
        if DEBUG:
            assign = ' '.join(f'{sym.arguments[0]}={sym.arguments[1]}' for sym in model.symbols(theory=True))
            print(f'  Model: {model}')
            print(f'  Assignment: {assign}')

    ctl.ground([('base', [])])
    thy.prepare(ctl)
    with ctl.solve(on_model=on_model, async_=True) as hnd:
        if not hnd.wait(finish_time - time.time()):
            hnd.cancel()
        res = hnd.get()
    if DEBUG:
        print(f'  Result: {res}\n')
    
    return res.unsatisfiable

def get_data(PATH):
    df = pd.read_csv(os.path.join(PATH, 'processed.csv'))
    correct_predictions = pd.read_csv(os.path.join(PATH, 'pred-index.csv'))
    correct_predictions = correct_predictions[correct_predictions.Pred==correct_predictions.target]
    index_correct_predictions = list(correct_predictions[correct_predictions.target==whichclass].level_0)
    input_var = np.array(df.columns[df.columns != target_variable])
    dom_of_each_var = [df[column].unique() for column in df.columns]

    return input_var, index_correct_predictions, dom_of_each_var, df

def load_neural_network(PATH):
    num_of_layers = 1  #number of hidden layers without input or output layer included
    num_of_neurons = [6,1] #Specify here the number of neurons in the hidden and the output layer

    weights = np.load(os.path.join(PATH,'weights.npy'), allow_pickle=True)
    biases = np.load(os.path.join(PATH,'biases.npy'), allow_pickle=True)

    return num_of_layers, num_of_neurons, weights, biases

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to explain neural network predictions using ASP.")  
    parser.add_argument('path', type=str, help="Path to the directory containing data and model files")  
    args = parser.parse_args() 
    PATH = args.path 

    input_var, index_correct_predictions, dom_of_each_var, df = get_data(PATH)
    num_of_layers, num_of_neurons, weights, biases = load_neural_network(PATH)

    #Neural network in the form of ASP encodings
    prg = ''
    prg += '%Create input variables\n'
    prg = asp_inputs(prg,dom_of_each_var,input_var)

    prg += '\n%Create formulas\n'
    for layer_idx in range(num_of_layers + 1):
        prg = asp_hidden(prg,np.transpose(weights[layer_idx]),biases[layer_idx],layer_idx,num_of_neurons[layer_idx],input_var)

    #Calculating explanations
    var_val_exp = []
    len_exp = []
    time_exp = []
    o = True if whichclass == 1 else False
    status = True
    start = time.time()

    for idx in index_correct_predictions:
        status = True
        local_start = time.time()
        W = set(input_var)
        V = df.loc[idx].to_dict()
        d = {}
        timeout = time.time() + TIMEOUT
        print(f'----------------------------Checking for {idx} ---------------------------------')
        for elem in input_var:
            print(W)
            W.remove(elem)
            result = nn_asp_explain(prg,o,W,V,timeout)
            if result is None:
                print('time out for index', idx)
                status = False
                break
            elif not result:
                print(f'Add {elem} to explanation set')
                W.add(elem)
                d[elem] = V[elem]
                status=True
        if status:
            local_end = time.time()
            var_val_exp.append(d)
            len_exp.append(len(W))
            time_exp.append(local_end-local_start)
            print(f'Elapsed time for calculating a local explanation{local_end-local_start}')


    if len(len_exp) > 0:
        end = time.time()
        print(len_exp)
        print(f'Avg len of an explanation {mean(len_exp)}')
        print(f'Elapsed time for calculating all explanations {end - start}')
        print(time_exp)
        print(f'Average time for one explanation {mean(time_exp)}')
        np.save(os.path.join(PATH, 'var_val_exp.npy'), np.array(var_val_exp, dtype=object), allow_pickle=True)


    
