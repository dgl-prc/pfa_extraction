from ObservationTable import ObservationTable
import DFA
from time import clock

def run_lstar(teacher,time_limit):
    table = ObservationTable(teacher.alphabet,teacher)
    start = clock()
    teacher.counterexample_generator.set_time_limit(time_limit,start)
    table.set_time_limit(time_limit,start)

    while True:
        # find an closed(? unclosed) table

        #DGL: get a stable OT
        while True:
            while table.find_and_handle_inconsistency():
                pass
            if table.find_and_close_row(): #returns whether table was unclosed
                continue
            else:
                break
        #DGL: after a stable OT got, then we build a DFA over the OT.
        dfa = DFA.DFA(obs_table=table)
        print("obs table refinement took " + str(int(1000*(clock()-start))/1000.0) )

        #DGL: we use the equivalence query to revise the model.
        counterexample = teacher.equivalence_query(dfa)
        if counterexample == None:
            break
        start = clock()
        # DGL: produce a new state and update the S.
        table.add_counterexample(counterexample,teacher.classify_word(counterexample))
    return dfa