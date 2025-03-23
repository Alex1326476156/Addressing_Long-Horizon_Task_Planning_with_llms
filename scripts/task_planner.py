import sys
import datetime
import httpx
import os

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
sys.path.append("virtualhome/simulation")
sys.path.append("virtualhome/simulation/unity_simulator")
sys.path.append("virtualhome/demo")
sys.path.append("virtualhome")

import argparse
import os.path as osp
from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from virtualhome.demo.utils_demo import *
import json
from utils_execute import *
import random
from virtualhome.demo.utils_demo import *
import openai
from openai import OpenAI
import urllib.request
import numpy as np
from virtualhome.simulation.evolving_graph import utils
from virtualhome.simulation.evolving_graph.scripts import parse_script_line, Script
from virtualhome.simulation.evolving_graph.execution import ScriptExecutor
from virtualhome.simulation.evolving_graph.environment import EnvironmentGraph
import time
import re
from utils_aug_env import get_obj_ids_for_adding_states, add_additional_obj_states

mode = 'manual'  # auto / manual
if mode == 'auto':
    if platform == 'darwin':
        exec_file = '../macos_exec*'
    else:
        exec_file = '../linux_exec*.x86_64'
    file_names = glob.glob(exec_file)
    if len(file_names) > 0:
        file_name = file_names[0]
        comm = UnityCommunication(file_name=file_name, port="8082", x_display="0")
    else:
        print("Error: executable path not found.")
else:
    comm = UnityCommunication()

# ---------------------------------------------------------------------------------------------------------------------
client = OpenAI(
    api_key="your_apikey"
)
# ---------------------------------------------------------------------------------------------------------------------
def LM(prompt,
       gpt_version,
       max_tokens=128,
       temperature=0,
       stop=None,
       logprobs=False,
       frequency_penalty=0):

    response = client.chat.completions.create(
        model=gpt_version,
        messages=[
            {"role": "system", "content": "Prompt:"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        logprobs=logprobs,
        frequency_penalty=frequency_penalty
    )

    if isinstance(response, list):
        generated_text = response[0].choices[0].message.content.strip()
    else:
        generated_text = response.choices[0].message.content.strip()

    return response, generated_text

def get_current_state_prompt():
    ## fixed function to define "PROMPT for state check"
    current_state_prompt = "kitchencounterdrawer, door is OPEN, character, wallpictureframe, clothespile is CLOSED, coffeemaker is OFF, pie, wall, bedroom, microwave is OFF and CLOSED, lightswitch is ON, kitchencabinet is CLOSED, washingsponge, bellpepper, salmon, fridge is CLOSED, wallshelf, tvstand, paper, floor, chips, photoframe, kitchen, whippedcream, candybar, faucet is OFF, tv is OFF, cereal, stovefan, waterglass, cutleryknife, kitchentable, condimentbottle, wineglass, bookshelf, cutleryfork, chocolatesyrup, walllamp, bench, sink, crackers, orchid, condimentshaker, kitchencounter is CLOSED, livingroom, powersocket, coffeepot is CLOSED, creamybuns, ceilinglamp, rug, book is CLOSED, plate, toaster is OFF, clock is OFF, wallphone is OFF, ceiling, fryingpan, box is CLOSED, dishbowl, bananas, breadslice, bathroom, garbagecan is CLOSED, stove is OFF and CLOSED, dishwashingliquid, plate ON kitchencounter, cutleryfork ON kitchentable, bookshelf ON floor, cutleryknife ON kitchentable, bellpepper ON kitchencounter, microwave ON kitchencounterdrawer, chocolatesyrup ON wallshelf, whippedcream ON rug, salmon ON microwave, orchid ON tvstand, wallpictureframe ON wall, bench ON floor, tvstand ON floor, book INSIDE bookshelf, bananas ON dishbowl, toaster ON kitchencounterdrawer, whippedcream ON kitchentable, dishbowl INSIDE bookshelf, fryingpan ON stove, rug ON kitchentable, coffeepot INSIDE coffeemaker, waterglass ON rug, dishwashingliquid ON kitchencounter, wallshelf ON wall, washingsponge ON kitchencounter, clothespile INSIDE bookshelf, bananas INSIDE bookshelf, box ON bookshelf, plate ON kitchentable, waterglass ON kitchentable, creamybuns ON wallshelf, breadslice INSIDE toaster, coffeemaker ON kitchencounterdrawer, chips ON wallshelf, book ON kitchentable, dishbowl ON bookshelf, pie ON kitchentable, wineglass ON tvstand, box ON tvstand, coffeepot ON kitchencounter, bellpepper ON kitchencounterdrawer, condimentshaker INSIDE bookshelf, coffeemaker ON kitchencounter, toaster ON kitchencounter, box INSIDE bookshelf, crackers ON wallshelf, character HOLD_RH book, faucet ON kitchencounter, book ON rug, cereal ON wallshelf, plate INSIDE microwave, candybar ON wallshelf, condimentbottle INSIDE bookshelf, tv ON tvstand, microwave ON kitchencounter, paper INSIDE bookshelf, kitchencounterdrawer ON kitchencounter, fridge ON floor, photoframe ON tvstand, wallpictureframe ON wallpictureframe, bench ON rug, pie ON rug, kitchencounterdrawer ON kitchencounterdrawer, dishbowl ON kitchencounter.\n\nassert('close' to 'mug' )\nFalse\nassert('close' to 'microwave' )\nTrue\nassert('book' is 'closed' )\nTrue\nassert('lightswitch' is 'OFF')\nFalse\nassert('book' in 'bookshelf')\nTrue\nassert('book' in 'hands')\nTrue\nassert('cereal' on 'bookshelf')\nFalse"
    objs = ['microwave', 'book', 'lightswitch', 'bookshelf', 'cereal']
    state, asserts = current_state_prompt = current_state_prompt.split('\n\n')
    state = state.split(',')
    state = "You see: " + ', '.join([i.strip() for i in state if any(element in i for element in objs)])
    current_state_prompt = f"{state}\n\n{asserts}"
    return current_state_prompt

current_state_prompt = get_current_state_prompt()

def run_execution(args, comm, test_tasks, gen_plan, log_file):
    final_states = [];
    initial_states = [];
    exec_per_task = []

    for task, plan in zip(test_tasks, gen_plan):
        ## initialize and set up enviroenment: visual + graph environment ##
        comm.reset(args.env_id)
        comm.add_character('Chars/Male2', initial_room='kitchen')

        _, graph = comm.environment_graph()
        _, cc = comm.camera_count()
        initial_states.append(graph)

        env_graph = EnvironmentGraph(graph)
        name_equivalence = utils.load_name_equivalence()
        executor = ScriptExecutor(env_graph, name_equivalence)

        ## get agent's initial state ##
        agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
        agent_in_roomid = \
        [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and n["relation_type"] == "INSIDE"][0]
        agent_in_room = [n['class_name'] for n in graph["nodes"] if n['id'] == agent_in_roomid][0]
        agent_has_objid = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and "HOLD" in n["relation_type"]]
        agent_has_obj = [n['class_name'] for n in graph["nodes"] if n['id'] in agent_has_objid]
        # some actions might not execute in the visual simulation, but they will in evolving graphs
        images = []
        max_fails = 10;
        num_fails = 0
        _, im = comm.camera_image([cc - 5], image_width=300, image_height=300)
        images.append(im[0])
        # s, obj = comm.get_visible_objects(cc-6)
        obj_ids_for_adding_states = get_obj_ids_for_adding_states(graph)
        nodes_with_additional_states = {}

        partial_graph = utils.get_visible_nodes(graph, agent_id=agent)

        obj_ids_close = [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and n["relation_type"] == "CLOSE"]
        obj = [node['class_name'] for node in partial_graph['nodes'] if node["id"] in obj_ids_close]
        obj_ids = dict([(node['id'], node['class_name']) for node in graph['nodes'] if
                        node["id"] in obj_ids_close and node['class_name'] in obj])
        relations = list(
            set([obj_ids[n['from_id']] + ' ' + n["relation_type"] + ' ' + obj_ids[n['to_id']] for n in graph["edges"] if
                 n['from_id'] in obj_ids and n['to_id'] in obj_ids and n["relation_type"] not in ["CLOSE", "FACING",
                                                                                                  "INSIDE"]]))
        obj_states = [(node['class_name'], node['states']) for node in graph['nodes'] if node['class_name'] in obj]
        objs = ""

        for ob_states in obj_states:
            if len(ob_states[1]) > 0:
                objs = objs + ob_states[0] + ' is ' + ' and '.join(ob_states[1]) + ', '
            else:
                objs = objs + ob_states[0] + ', '
        objs = list(set(objs.split(', ')))
        objs = [ob for ob in objs if len(ob) > 0]
        objs = ', '.join(objs) + ', ' + ', '.join(relations) + '. '
        if len(agent_has_obj) > 0:
            agent_has_obj = ', '.join(agent_has_obj)
            objs += f" You have {agent_has_obj}. "

        ## parse plan into subgoals ##
        log_file.write(f"\n--Executing task: {task}--\n")
        log_file.write(f"Plan:  {plan}\n\n")
        print(f"Executing: {task}\n")

        subgoals = {}
        subgoals['0'] = []
        sg = None

        for i in plan.split('\n'):
            i = i.strip()
            if len(i) < 1:
                continue
            if "comments" in args.prompt_task_examples_ablation:
                subgoals['0'].append(i)
            else:
                if "#" in i:
                    sg = i.split("#")[1].strip()
                    subgoals[sg] = []
                else:
                    if sg is not None:
                        subgoals[sg].append(i)

        ## begin execution ##
        executable_steps = 0;
        total_steps = 0
        last_assert = None
        for subgoal in subgoals.keys():
            step = 1;
            act = ""
            for action in subgoals[subgoal]:
                # fixes needed for not getting stuck
                if step > 10:
                    break
                if "grab('wallphone')" in action:
                    continue

                ## state checking ##

                # parse asserts and query LLM
                if "assert" in action:
                    check_state = "";
                    last_assert = action
                    assert_objs = re.findall(r"\b[a-z]+", action)[1::2]
                    state = objs.split(',')
                    state = "You see: " + ', '.join([i.strip() for i in state if any(ele in i for ele in assert_objs)])
                    current_state = f"{current_state_prompt}\n\n{state}\n\n{action}\n"
                    _, check_state = LM(current_state, args.gpt_version,
                                        max_tokens=2, stop=["\n"])
                    # time.sleep(19)
                    log_file.write(f"State check:\n{state}\n{action}\n{check_state.strip()}\n")
                    continue

                # get recovery actions
                if last_assert != None:
                    if "True" in check_state:
                        # skip revovery if state check is true
                        if "else: " in action:
                            continue
                    elif "False" in check_state:
                        if "else: " in action:
                            action = action.split(': ')[-1].strip()
                        else:
                            state = objs.split(',')
                            state = "You see: " + ', '.join(
                                [i.strip() for i in state if any(ele in i for ele in assert_objs)])
                            current_state = f"{current_state_prompt}\n\n{state}\n\n{action}\n"
                            _, check_state = LM(current_state, args.gpt_version,
                                                max_tokens=2, stop=["\n"])
                            # time.sleep(19)
                            log_file.write(f"State check:\n{state}\n{action}\n{check_state.strip()}\n")

                # since above steps are not for env, following line go through the env
                total_steps += 1

                ## parse next action
                action = action.split(')')[0]
                action = re.findall(r"\b[a-z]+", action)
                found_id = None

                if len(action) == 3 and "put" in action[0]:  # 2 objs action
                    obj_id1 = [node['id'] for node in graph['nodes'] if
                               node['class_name'] == action[1] and node['id'] in agent_has_objid]
                    obj_id2 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[2]]
                    if len(obj_id1) == 0:
                        step + 1;
                        log_file.write("obj not in hand\n");
                        continue
                    if len(obj_id1) == 1:
                        id1 = obj_id1[0]
                    else:
                        id1 = random.choice(obj_id1)

                    if len(obj_id2) == 0:
                        step + 1;
                        log_file.write("obj not found\n");
                        continue
                    elif len(obj_id2) == 1:
                        id2 = obj_id2[0]
                    elif found_id in obj_id2:
                        id2 = found_id
                    else:
                        id2 = random.choice(obj_id2)
                    script_instruction = '<char0> [{}] <{}> ({}) <{}> ({})'.format(action[0], action[1], id1, action[2],
                                                                                   id2)
                elif len(action) == 2 and action[0] not in ["find", "walk"]:  # 1 obj action
                    obj_id1 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[1]]
                    if len(obj_id1) == 1:
                        id1 = obj_id1[0]
                    elif found_id in obj_id1:
                        id1 = found_id
                    elif len(obj_id1) == 0:
                        step + 1;
                        log_file.write("obj not found\n");
                        continue
                    else:
                        id1 = random.choice(obj_id1)
                    script_instruction = '<char0> [{}] <{}> ({})'.format(action[0], action[1], id1)
                elif len(action) == 2:  # walk or find action
                    obj_id1 = [node['id'] for node in graph['nodes'] if node['class_name'] == action[1]]
                    if len(obj_id1) == 0:
                        step + 1;
                        log_file.write("obj not found\n");
                        continue
                    found_id = random.choice(obj_id1)
                    script_instruction = '<char0> [{}] <{}> ({})'.format(action[0], action[1], found_id)
                elif len(action) == 1:  # 0 object action
                    script_instruction = '<char0> [{}]'.format(action[0])
                else:
                    log_file.write("bad action\n");
                    continue

                ## execute next action in both envs: visual and graph
                log_file.write(f"{script_instruction}\n")
                _, m = comm.render_script([script_instruction], recording=True, find_solution=True)
                script = script_instruction[7:]
                try:
                    script = parse_script_line(script, 0)
                except:
                    step += 1;
                    continue
                print(script)
                success, final_state, _ = executor.execute(Script([script]))

                if not success:
                    log_file.write(f"act_success: {success}, message: {executor.info.get_error_string()}\n")
                    step += 1
                else:
                    # count execution if action executes succesfully in graph env
                    executable_steps += 1
                    # _, graph = comm.environment_graph()
                    graph = final_state.to_dict()
                    env_graph = EnvironmentGraph(graph)
                    agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
                    partial_graph = utils.get_visible_nodes(final_state.to_dict(), agent_id=agent)
                    name_equivalence = utils.load_name_equivalence()
                    executor = ScriptExecutor(env_graph, name_equivalence)
                    script_instruction = ' '.join(re.findall(r"\b[a-z]+", script_instruction)[1:])
                    step += 1

                    # get new state info
                    agent = [n['id'] for n in graph["nodes"] if n['class_name'] == 'character'][0]
                    agent_in_roomid = \
                    [n['to_id'] for n in graph["edges"] if n['from_id'] == agent and n["relation_type"] == "INSIDE"][0]
                    agent_in_room = [n['class_name'] for n in graph["nodes"] if n['id'] == agent_in_roomid][0]
                    agent_has_objid = [n['to_id'] for n in graph["edges"] if
                                       n['from_id'] == agent and "HOLD" in n["relation_type"]]
                    agent_has_obj = [n['class_name'] for n in graph["nodes"] if n['id'] in agent_has_objid]

                    # Here you can get an observation, for instance
                    if 'grab' in script_instruction or 'open' in script_instruction or 'close' in script_instruction:
                        s, im = comm.camera_image([cc - 5], image_width=300, image_height=300)
                    else:
                        s, im = comm.camera_image([cc - 6], image_width=300, image_height=300)
                    images.append(im[0])

                    obj_ids_close = [n['to_id'] for n in graph["edges"] if
                                     n['from_id'] == agent and n["relation_type"] == "CLOSE"]
                    obj = [node['class_name'] for node in partial_graph['nodes'] if node["id"] in obj_ids_close]
                    obj_ids = dict([(node['id'], node['class_name']) for node in partial_graph['nodes'] if
                                    node["id"] in obj_ids_close and node['class_name'] != agent_in_room])
                    nodes_with_additional_states = add_additional_obj_states(partial_graph, obj_ids_for_adding_states,
                                                                             nodes_with_additional_states)

                    relations = list(
                        set([obj_ids[n['from_id']] + ' ' + n["relation_type"] + ' ' + obj_ids[n['to_id']] for n in
                             graph["edges"] if
                             n['from_id'] in obj_ids and n['to_id'] in obj_ids and n["relation_type"] not in ["CLOSE",
                                                                                                              "FACING"]]))
                    obj_states = [(node['class_name'], node['states']) for node in graph['nodes'] if
                                  node['class_name'] in obj]
                    objs = ""
                    for ob_states in obj_states:
                        if len(ob_states[1]) > 0:
                            objs = objs + ob_states[0] + ' is ' + ' and '.join(ob_states[1]) + ', '
                        else:
                            objs = objs + ob_states[0] + ', '
                    objs = list(set(objs.split(', ')))
                    objs = [ob for ob in objs if len(ob) > 0]
                    objs = ', '.join(objs) + ', ' + ', '.join(relations) + '. '

                    if len(agent_has_obj) > 0:
                        agent_has_obj = ', '.join(agent_has_obj)
                        objs += f" You have {agent_has_obj}. "

        if final_state is not None:
            if isinstance(final_state, dict):
                if "nodes" in final_state.keys():
                    for idx in range(len(final_state["nodes"])):
                        if final_state["nodes"][idx]['id'] in nodes_with_additional_states.keys():
                            final_state["nodes"][idx] = nodes_with_additional_states[final_state["nodes"][idx]['id']]
            else:
                final_state = final_state.to_dict()
                for idx in range(len(final_state["nodes"])):
                    if final_state["nodes"][idx]['id'] in nodes_with_additional_states.keys():
                        final_state["nodes"][idx] = nodes_with_additional_states[final_state["nodes"][idx]['id']]

            # get final state for eval
            final_states.append(final_state)
        else:
            print("Error: final_state is not set, skipping appending to final_states.")

        exec_per_task.append(executable_steps / total_steps)

    return final_states, initial_states, exec_per_task


def eval(final_states,
         final_states_GT,
         initial_states,
         test_tasks,
         exec_per_task,
         log_file):

    sr = []
    unsatif_conds = [];
    unchanged_conds = []
    total_goal_conds = [];
    total_unchanged_conds = []
    results = {}
    for g, g_gt, g_in, d in zip(final_states, final_states_GT, initial_states, test_tasks):
        obj_ids = dict([(node['id'], node['class_name']) for node in g_in['nodes']])
        relations_in = set(
            [obj_ids[n['from_id']] + ' ' + n["relation_type"] + ' ' + obj_ids[n['to_id']] for n in g_in["edges"]])
        obj_states_in = set([node['class_name'] + ' ' + st for node in g_in['nodes'] for st in node['states']])

        obj_ids = dict([(node['id'], node['class_name']) for node in g['nodes']])
        relations = set(
            [obj_ids[n['from_id']] + ' ' + n["relation_type"] + ' ' + obj_ids[n['to_id']] for n in g["edges"]])
        obj_states = set([node['class_name'] + ' ' + st for node in g['nodes'] for st in node['states']])

        obj_ids = dict([(node['id'], node['class_name']) for node in g_gt['nodes']])
        relations_gt = set(
            [obj_ids[n['from_id']] + ' ' + n["relation_type"] + ' ' + obj_ids[n['to_id']] for n in g_gt["edges"]])
        obj_states_gt = set([node['class_name'] + ' ' + st for node in g_gt['nodes'] for st in node['states']])


        #wyyd
        relations_gt_condition = relations_gt - relations_in
        relations_condition = relations - relations_in
        obj_states_gt_condition = obj_states_gt - obj_states_in
        obj_states_condition = obj_states - obj_states_in


        print(len((relations_gt - relations_in))) #
        print(len((relations - relations_in)))
        print(len((obj_states_gt - obj_states_in)))
        print(len((obj_states - obj_states_in)))


        log_file.write(
            f"\nunsatisfied state conditions: relations: {(relations_gt - relations_in) - (relations - relations_in)}, object states: {(obj_states_gt - obj_states_in) - (obj_states - obj_states_in)}")
        unsatif_conds.append((len((relations_gt - relations_in) - (relations - relations_in)) + len(
            (obj_states_gt - obj_states_in) - (obj_states - obj_states_in))))
        total_goal_conds.append(len(relations_gt - relations_in) + len(obj_states_gt - obj_states_in))
        sr.append(1 - unsatif_conds[-1] / total_goal_conds[-1])

        unchanged_conds.append((len(relations_gt.intersection(relations_in) - relations) + len(
            obj_states_gt.intersection(obj_states_in) - obj_states)))
        total_unchanged_conds.append(
            len(relations_gt.intersection(relations_in)) + len(obj_states_gt.intersection(obj_states_in)))

        results[d] = {'PSR': sr[-1],
                      "SR": sr[-1:].count(1.0),
                      "Precision": 1 - unchanged_conds[-1] / total_unchanged_conds[-1],
                      "Exec": exec_per_task[-1]
                      }

    results["overall"] = {'PSR': sum(sr) / len(sr),
                          "SR": sr.count(1.0) / len(sr),
                          "Precision": 1 - sum(unchanged_conds) / sum(total_unchanged_conds),
                          "Exec": sum(exec_per_task) / len(exec_per_task)
                          }
    return results


def planner_executer(args):
    # initialize env

    comm.reset(0)

    _, env_graph = comm.environment_graph()
    obj = list(set([node['class_name'] for node in env_graph["nodes"]]))

    prompt = f"from actions import turnright, turnleft, walkforward, walktowards <obj>, walk <obj>, run <obj>, grab <obj>, switchon <obj>, switchoff <obj>, open <obj>, close <obj>, lookat <obj>, sit <obj>, standup, find <obj>, turnto <obj>, drink <obj>, pointat <obj>, watch <obj>, putin <obj> <obj>, putback <obj> <obj>"

    #prompt += defined_actions
    #prompt += f"\n\nobjects = {obj}\n\t"
    #prompt += object_property_prompt
    #prompt += user_prompt

    # load train split for task examples
    with open(f"{args.progprompt_path}/data/pythonic_plans/train_complete_plan_set.json", 'r') as f:
        tmp = json.load(f)
        prompt_egs = {}
        for k, v in tmp.items():
            prompt_egs[k] = v

    # default examples from the paper
    if args.prompt_task_examples == "default":
        default_examples = ["put_the_wine_glass_in_the_kitchen_cabinet",
                            "throw_away_the_lime",
                            "wash_mug",
                            "refrigerate_the_salmon",
                            "bring_me_some_fruit",
                            "wash_clothes",
                            "put_apple_in_fridge"]
        for i in range(args.prompt_num_examples):
            prompt += "\n\n" + prompt_egs[default_examples[i]] + "\n\n"


    # random egs - change seeds
    if args.prompt_task_examples == "random":
        random.seed(args.seed)
        prompt_egs_keys = random.sample(list(prompt_egs.keys()), args.prompt_num_examples)

        for eg in prompt_egs_keys:
            prompt += "\n\n" + prompt_egs[eg]

    # evaluate in given unseen env
    if args.env_id != 0:
        #print(args.env_id)
        comm.reset(args.env_id)
        comm.reset(args.env_id)
        _, graph = comm.environment_graph()
        obj = list(set([node['class_name'] for node in graph["nodes"]]))
        prompt += f"\n\n\nobjects = {obj}"

        # evaluation tasks in given unseen env
        test_tasks = []
        with open(f"{args.progprompt_path}/data/new_env/{args.test_set}_annotated.json", 'r') as f:
            for line in f.readlines():
                test_tasks.append(list(json.loads(line).keys())[0])

    # setup logging
    log_filename = f"{args.expt_name}_{args.prompt_task_examples}_{args.prompt_num_examples}examples"
    if args.prompt_task_examples_ablation != "none":
        log_filename += f"_{args.prompt_task_examples_ablation}"
    log_filename += f"_{args.test_set}"
    log_file = open(f"{args.progprompt_path}/results/{log_filename}_logs.txt", 'w')
    log_file.write(f"\n----PROMPT for planning----\n{prompt}\n")
    print("prompt generating......")

    # evaluate in seen env
    if args.env_id == 0:
        # print("args.env_id = ", args.env_id)
        test_tasks = []
        for file in os.listdir(f"{args.progprompt_path}/data/{args.test_set}"):
            with open(f"{args.progprompt_path}/data/{args.test_set}/{file}", 'r') as f:
                for line in f.readlines():
                    test_tasks.append(list(json.loads(line).keys())[0])

        log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
        print("Test set tasks generating......")

    # test_tasks = test_tasks[:3] ## debug to check changes

    # generate plans for the test set
    print("here is args.load_generated_plans", args.load_generated_plans)
    if not args.load_generated_plans:
        test_tasks = ["watch_tv"]
        gen_plan = [
            """
            def watch_tv():
            # 0: Walk to the living room
            walk('livingroom')
            
            # 1: Find the TV
            find('tv')
            
            # 2: Turn on the TV if it is off
            assert('close' to 'tv')
            else: find('tv')
            assert('tv' is 'switchoff')
            else: switchoff('tv')
            switchon('tv')
            
            # 3: Watch the TV
            assert('close' to 'tv')
            else: find('tv')
            watch('tv')
            
            # 4: Done
            """
        ]
        # save generated plan
        line = {}
        # print(f"Saving generated plan at: {log_filename}_plans.json\n")

        current_time = datetime.datetime.now()
        print("Current system time:", current_time)


        for plan, task in zip(gen_plan, test_tasks):
            line[task] = plan


        with open("generated_plans.json", 'w') as f:
            json.dump(line, f, ensure_ascii=False, indent=4)

        print("Generated plans saved to 'generated_plans.json'")

    # load from file
    else:
        print(f"Loading generated plan from: {log_filename}.json\n")
        print("loading generated plan ok")

        file_path = "E:\\experiment\\demo\\progprompt-vh-main\\results\\expt030_default_3examples_test_unseen_plans.json"

        # if os.path.exists(file_path):
        #     with open(file_path, 'r') as f:
        #         content = f.read()
        #         print("file exists")
        # else:
        #     print("No file")

        with open(f"E:\experiment\demo\progprompt-vh-main/results/{log_filename}_plans.json", 'r') as f:
            print("open file ok")
            data = json.load(f)
            test_tasks, gen_plan = [], []
            for k, v in data.items():
                test_tasks.append(k)
                gen_plan.append(v)

    log_file.write(f"\n----PROMPT for state check----\n{current_state_prompt}\n")

    # run execution

    print(f"\n----Runing execution----\n")
    final_states, initial_states, exec_per_task = run_execution(args,
                                                                comm,
                                                                test_tasks,
                                                                gen_plan,
                                                                log_file)

    # evaluate
    final_states_GT = []
    with open(f'{args.progprompt_path}/data/final_states/final_states_{args.test_set}.json', 'r') as f:
        for line in f.readlines():
            final_states_GT.append((json.loads(line)))

    results = eval(final_states,
                   final_states_GT,
                   initial_states,
                   test_tasks,
                   exec_per_task,
                   log_file)

    print(f"\n----Results----\n{results['overall']}\n")
    with open(f"{args.progprompt_path}/results/{log_filename}_metric.json", 'w') as f:
        json.dump(results, f)
    log_file.close()
    print("log_file close done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--progprompt-path", type=str,
                        default="")
    parser.add_argument("--expt-name", type=str,
                        default="")

    parser.add_argument("--openai-api-key", type=str,
                        default="")
    parser.add_argument("--unity-filename", type=str,
                        default="")
    parser.add_argument("--port", type=str, default="8082")
    parser.add_argument("--display", type=str, default="0")

    parser.add_argument("--gpt-version", type=str, default="gpt-3.5-turbo",
                        choices=['gpt-3.5-turbo', 'davinci-002', 'babbage-002'])
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--test-set", type=str, default="test_unseen",
                        choices=['test_unseen', 'test_seen', 'test_unseen_ambiguous', 'env1', 'env2'])

    parser.add_argument("--prompt-task-examples", type=str, default="default",
                        choices=['default', 'random'])

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--prompt-num-examples", type=int, default=3,
                        choices=range(1, 7))
    parser.add_argument("--prompt-task-examples-ablation", type=str, default="none",
                        choices=['none', 'no_comments', "no_feedback", "no_comments_feedback"])

    parser.add_argument("--load-generated-plans", type=bool, default=False)

    args, unknown = parser.parse_known_args()

    openai.api_key = args.openai_api_key

    if not osp.isdir(f""):
        os.makedirs(f"")

    planner_executer(args=args)
