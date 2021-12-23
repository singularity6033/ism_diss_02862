# for given recall and precision calculate the Average Precision (AP)
def compute_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    m_rec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    m_pre = prec[:]
    # i: len(m_pre)-2~0 (if the right hand side is 0, i will end with 1)
    # replace each precision value with the maximum precision value to the right of that recall level
    # make the zig-zag pattern change to decrease monotonically
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    # creates a list of index where the recall changes
    i_list = []
    for i in range(1, len(m_rec)):
        if m_rec[i] != m_rec[i - 1]:
            i_list.append(i)
    # AP is the area under the curve ()
    ap = 0.0
    for i in i_list:
        ap += ((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return ap

# recall = [0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 1]
# precision = [1, 1, 0.67, 0.5, 0.4, 0.5, 0.57, 0.5, 0.44, 0.5]
# ap = ap_compute(recall, precision)
# print(ap)
