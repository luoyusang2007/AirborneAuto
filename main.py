
'''

'''
from enum import IntEnum
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np

# States of an individual
class State(IntEnum):
    Healthy = 0  # Blue
    Incubation = 4  # Purple
    Light_Symptom = 5  # brown
    Heavy_Symptom = 3  # Red
    Cured_Dead_or_Isolated = 7  # Grey

GVAR_statistic = {
    State.Healthy:[],
    State.Incubation:[],
    State.Light_Symptom:[],
    State.Heavy_Symptom:[],
    State.Cured_Dead_or_Isolated:[]
}

# 常量
GVAR_MAX_DISTANCE = 2  # 距离大于GVAR_MAX_DISTANCE不会被传染
GVAR_MAX_TIME = 15  # 任何一个病程不会持续GVAR_MAX_TIME天以上

# 【尚未实现】家庭分割
GVAR_ROW_GRID = 3
GVAR_COL_GRID = 1

# Canvas Size/仿真画布大小
GVAR_ROWS = 64
GVAR_COLS = 64

# Variables/变量
GVAR_time = 0
GVAR_data_2d = np.zeros((GVAR_ROWS, GVAR_COLS), dtype=np.uint8)
GVAR_start_time_2d = np.zeros((GVAR_ROWS, GVAR_COLS), dtype=np.int32)

# 【尚未实现】间歇性外出（距离）
GVAR_ChanceOutdoorActivities_vsDistance={
    State.Healthy:[0.01, 0.01, 0, 0, 0, 0.01],
    State.Incubation:[0.01, 0.01, 0, 0, 0, 0.01],
    State.Light_Symptom:[0.005, 0.005, 0, 0, 0, 0.005],
    State.Heavy_Symptom:[0.005, 0.005, 0, 0, 0, 0.005]
}


# Chance of a healthy person to be infected vs Diatance/一个健康人被周围患者感染的概率（随距离变化）
GVAR_ChanceHealthy_InfectByInfected_vsDistance = {
    State.Incubation: [0.2, 0.1,0.1],  # 单位距离
    State.Light_Symptom: [0.4, 0.2,0.2],
    State.Heavy_Symptom: [0.4, 0.2,0.2]
}

# 状态转移
GVAR_ChanceStateChange_vsTime = {
    State.Incubation:{# 一个潜伏期患者转换为其它状态的概率（随时间变化）
        State.Light_Symptom:[0, 0, 0, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
        "default":State.Cured_Dead_or_Isolated
    },
    State.Light_Symptom:{# 一个轻症患者转换为其它状态的概率（随时间变化）
        State.Heavy_Symptom:[0.05, 0.05, 0.1, 0.1, 0.1, 0.05],
        State.Cured_Dead_or_Isolated:[0.05, 0.05, 0.05, 0.05, 0.05, 0.005],
        "default":State.Cured_Dead_or_Isolated
    },
    State.Heavy_Symptom:{# 一个重症患者转换为其它状态的概率（随时间变化）
        State.Cured_Dead_or_Isolated:[0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "default":State.Cured_Dead_or_Isolated
    }
}


def set_max_len(dict_of_lists):
    max_length = 0
    for k,list_v in dict_of_lists.items():
        if k in ["default","max_length"]:
            continue
        max_length = len(list_v) if len(list_v)>max_length else max_length
    dict_of_lists["max_length"] = max_length

def format_state_change_dict(state_change_dict):
    for k_form, dict_target_state in state_change_dict.items():
        if dict_target_state.get(k_form):
            raise Exception("Self can not be included!")
        dict_target_state[k_form] = [1]*dict_target_state["max_length"]
        for k_target_state, p_list in dict_target_state.items():
            if k_target_state in ["default","max_length",k_form]:
                continue
            for col in range(dict_target_state["max_length"]):
                dict_target_state[k_form][col] = dict_target_state[k_form][col] - p_list[col]
            
    # 补足长度
    # 给出自身

def ani_run(data):
    update_interval = 50
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(data, cmap="tab10",
                    interpolation='nearest', vmin=0, vmax=9)

    # 以下变量要保留
    ani_ = ani.FuncAnimation(
        fig = fig, 
        func = generate_frame,
        fargs=(img, plt, data),
        frames=60,
        interval=update_interval,
        save_count=50
    )
    # ani_.save('spread.gif',writer='imagemagick')
    plt.show()


def generate_frame(frame_num, img, plt, initial):
    global GVAR_time
    GVAR_statistic[State.Healthy].append(0)
    GVAR_statistic[State.Incubation].append(0)
    GVAR_statistic[State.Light_Symptom].append(0)
    GVAR_statistic[State.Heavy_Symptom].append(0)
    GVAR_statistic[State.Cured_Dead_or_Isolated].append(0)

    plt.title('{GVAR_time} Timesteps'.format(GVAR_time=GVAR_time))
    data = initial.copy()
    for row in range(GVAR_ROWS):
        for col in range(GVAR_COLS):
            prev_state = initial[row, col]
            GVAR_statistic[prev_state][-1] += 1 
            time_passed = GVAR_time-GVAR_start_time_2d[row, col]

            # 如果这个人没被感染，计算他被感染的概率
            if prev_state == State.Healthy:
                # 没被传染的概率
                pn_inf = 1

                # 看周围的人，计算没被传染的概率
                for _row in range(row-1, row+2):
                    for _col in range(col-1, col + 2):
                        if (
                            _row in range(GVAR_ROWS) and # 不超过画布边界
                            _col in range(GVAR_COLS) and # 不超过画布边界
                            initial[_row, _col] in [
                                State.Incubation, 
                                State.Light_Symptom, 
                                State.Heavy_Symptom
                            ] and # 已被感染的人才能传染给你
                            (_row != row or _col != col)# 自己不能传染自己
                        ): 
                            dr = row - _row
                            dc = col - _col
                            distance = max(abs(dr), abs(dc))  # 为了速度
                            pn_inf *= 1 - GVAR_ChanceHealthy_InfectByInfected_vsDistance[initial[_row, _col]][distance]

                data[row, col] = np.random.choice(
                    [State.Incubation, State.Healthy], p=[1-pn_inf, pn_inf]
                )
            
            # 以下几种情况都是这个人已经被感染了，病程会发展
            elif prev_state == State.Incubation:
                if time_passed < GVAR_ChanceStateChange_vsTime[prev_state]["max_length"]:
                    p_dict = {k:p_list[time_passed] for k,p_list in GVAR_ChanceStateChange_vsTime[prev_state].items() if k not in ["max_length", "default"]}
                    data[row, col] = np.random.choice(
                        list(p_dict.keys()),
                        p=list(p_dict.values())
                    )
                else:
                    data[row, col] = GVAR_ChanceStateChange_vsTime[prev_state]["default"]

            elif prev_state == State.Light_Symptom:
                if time_passed < GVAR_ChanceStateChange_vsTime[prev_state]["max_length"]:
                    p_dict = {k:p_list[time_passed] for k,p_list in GVAR_ChanceStateChange_vsTime[prev_state].items() if k not in ["max_length", "default"]}
                    data[row, col] = np.random.choice(
                        list(p_dict.keys()),
                        p=list(p_dict.values())
                    )
                else:
                    data[row, col] = GVAR_ChanceStateChange_vsTime[prev_state]["default"]

            elif prev_state == State.Heavy_Symptom:
                if time_passed < GVAR_ChanceStateChange_vsTime[prev_state]["max_length"]:
                    p_dict = {k:p_list[time_passed] for k,p_list in GVAR_ChanceStateChange_vsTime[prev_state].items() if k not in ["max_length", "default"]}
                    data[row, col] = np.random.choice(
                        list(p_dict.keys()),
                        p=list(p_dict.values())
                    )
                else:
                    data[row, col] = GVAR_ChanceStateChange_vsTime[prev_state]["default"]


            # 被隔离的/死了的/治愈了的不会再被感染了。
            elif prev_state == State.Cured_Dead_or_Isolated:
                pass
            else:
                print("Err: previous state:", prev_state)
                raise Exception

            # 如果这个人病情变化了，记录进入新状态的时间
            if data[row, col] != prev_state:
                print("Individual({col},{row}) become {new_state} after {time_passed} timesteps at time {time_now}".format(
                    col=col,
                    row=row,
                    new_state=data[row, col],
                    time_passed=time_passed,
                    time_now=GVAR_time
                ))
                GVAR_start_time_2d[row, col] = GVAR_time
    img.set_data(data)
    initial[:] = data[:]
    GVAR_time += 1
    return img

if __name__ == "__main__":
    # Set Max:
    set_max_len(GVAR_ChanceHealthy_InfectByInfected_vsDistance)
    for v_dict in GVAR_ChanceStateChange_vsTime.values():
        set_max_len(v_dict)

    format_state_change_dict(GVAR_ChanceStateChange_vsTime)



    GVAR_data_2d[15, 20] = State.Incubation
    ani_run(GVAR_data_2d)
