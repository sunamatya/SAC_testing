import pickle
#import pickle5 as pickle
import sys
import csv
import numpy as np
sys.path.append("C:\\Users\\samatya.ASURITE\\PycharmProjects\\SocialGracefullnessTIV")


header = ['it', 'eps', 'distance', 'sx', 'sy', 'intent_loss_1', 'intent_loss_2', 'collsion_loss', 'Value']

with open('AV_outdata_s_v_g_intermittent.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

#data_initial = pickle.load(open("uniform_data_dist_updated.p", "rb"))

for i in range(751, 1000): #range(751):
    #output_filename = "sim_output/variable_start_first_30/data_trial_" +str(i)+"/output.pkl"
    #output_filename = "sim_outputs_empathetic/data_trial_" + str(i) + "/output.pkl"
    output_filename = "correct_res_out/correct_res_out/data_trial_" + str(i) + "/output.pkl"
    data = pickle.load(open(output_filename, "rb"))

    for j in range(0, len(data.car1_states)-2):
        distance = np.sqrt(data.car1_states[j+1][0]* data.car1_states[j+1][0] + data.car2_states[j+1][1]* data.car2_states[j+1][1])
        sx = data.car1_states[j+1][0]
        sy = data.car2_states[j+1][1]
        #gracefulness = data.car1_gracefulness[j]
        intent_loss_1 = data.car1_planned_loss[j] - data.car1_collision_loss[j]
        intent_loss_2 = data.car2_planned_loss[j] - data.car1_collision_loss[j]
        collision_loss = data.car1_collision_loss[j]
        inference_cost = 400 * data.car1_does_inference[j]
        value = -(data.car1_planned_loss[j] + (data.car2_planned_loss[j] - data.car1_collision_loss[j])/1e3 + inference_cost)
        TX1 = data.car1_states[j+1][0] / data.car1_actions[j+1][0]
        TX2 = data.car2_states[j+1][1]/ data.car2_actions[j+1][1]
        TTC = TX2-TX1



        data_row = [i, j, distance, sx, sy, intent_loss_1, intent_loss_2, collision_loss, value]
        with open('AV_outdata_s_v_g_intermittent.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(header)
            writer.writerow(data_row)
