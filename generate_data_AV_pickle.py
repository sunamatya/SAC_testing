import pickle
import sys
import csv
sys.path.append("C:\\Users\\samatya.ASURITE\\PycharmProjects\\SocialGracefullnessTIV")

# time, states, agg, intent"
header = ['ts', 'sx', 'sy', 'vx', 'vy', 'ui', 'uj', 'bji1', 'bji2', 'bji3', 'bji4', 'bij1', 'bij2', 'bij3', 'bij4', 'agg', 'infc1', 'infc2']

with open('AV_outdata_no_error.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

data_initial = pickle.load(open("uniform_data_dist_updated.p", "rb"))

for i in range(151, 201): #range(751):
    #output_filename = "sim_output/variable_start_first_30/data_trial_" +str(i)+"/output.pkl"
    #output_filename = "correct_res_out/correct_res_out/data_trial_" + str(i) + "/output.pkl"
    #output_filename = "random_200_var_cost_function_half/data_trial_" + str(i) + "/output.pkl"
    #output_filename = "ability/data_trial_" + str(i) + "/output.pkl"
    output_filename = "no_error/data_trial_" + str(i) + "/output.pkl"
    #output_filename = "output.pkl"
    bji_1 = data_initial["bji"][i][0]
    bji_2 = data_initial["bji"][i][1]
    bji_3 = data_initial["bji"][i][2]
    bji_4 = data_initial["bji"][i][3]

    bij_1 = data_initial["bij"][i][0]
    bij_2 = data_initial["bij"][i][1]
    bij_3 = data_initial["bij"][i][2]
    bij_4 = data_initial["bij"][i][3]

    agg = data_initial["agg"][i]

    # if agg:
    #     aggr = 1 #1e6
    # else:
    #     aggr = 0 #

    aggr = 1e6
    data = pickle.load(open(output_filename, "rb"))


    data_row = [0, data_initial["si"][i], data_initial["sj"][i], 0.025, -0.025, 0, 0, bji_1, bji_2, bji_3, bji_4, bij_1,
                bij_2, bij_3, bij_4, aggr,  data.car1_does_inference[0],  data.car2_does_inference[0]]

    with open('AV_outdata_no_error.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        #writer.writerow(header)
        writer.writerow(data_row)

    for j in range(1, len(data.car1_states)-1):
        bji_1 = data.car1_joint_probability_matrix[j-1][0][0]
        bji_2 = data.car1_joint_probability_matrix[j-1][0][1]
        bji_3 = data.car1_joint_probability_matrix[j-1][1][0]
        bji_4 = data.car1_joint_probability_matrix[j-1][1][1]

        bij_1 = data.car2_joint_probability_matrix[j-1][0][0]
        bij_2 = data.car2_joint_probability_matrix[j-1][0][1]
        bij_3 = data.car2_joint_probability_matrix[j-1][1][0]
        bij_4 = data.car2_joint_probability_matrix[j-1][1][1]

        # if j > 0 and j % 15 == 0:
        #     if aggr == 1:
        #         aggr = 1e6
        #     else:
        #         aggr = 1

        data_row = [j, data.car1_states[j][0], data.car2_states[j][1], data.car1_actions[j][0], data.car2_actions[j][1],
                    data.car1_planned_trajectory_set[j-1][0], data.car2_planned_trajectory_set[j-1][0],
                    bji_1, bji_2, bji_3, bji_4, bij_1, bij_2, bij_3, bij_4, aggr, data.car1_does_inference[j], data.car2_does_inference[j]]

        with open('AV_outdata_no_error.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(header)
            writer.writerow(data_row)
    #sim_data = Sim_Data()
    # print(len(data.car1_states))
    # print(len(data.car1_actions))
    # print(len(data.car1_does_inference))


    # write the header


    # write the data
