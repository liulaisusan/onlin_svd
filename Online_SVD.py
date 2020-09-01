import numpy as np


def increment_svd(U, S, V, A, B):
    # print(U)
    # print(S)
    # print(V)
    # print(A)
    # print(B)
    u_x, u_y = U.shape # d_u <- dim(U),
    rank_u = u_y # rankU <- d_u[2]
    a_x, a_y = A.shape # d_a <- dim(A)

    ua = np.concatenate((U, A), axis=1) # cBind(U,A)
    qp,rp = np.linalg.qr(ua, mode='complete') # QP <- Matrix(qr.Q(qrP, complete = TRUE), sparse = TRUE) RP <- Matrix(qr.R(qrP, complete = TRUE), sparse = TRUE)
    # w = u_y + 1 # w <- rankU+1
    qp_x, qp_y = qp.shape # d_qp <- dim(QP)
    rp_x, rp_y = rp.shape # d_rp <- dim(RP)
    # print(qp)
    # print(rp)

    if qp_y == rank_u:
        P = qp[:,rank_u : qp_y]
        Ra = rp[rank_u:rp_x , rank_u:rp_y].T
        dim_Ra = 0
    else:
        P = qp[:,rank_u : qp_y]
        Ra = rp[rank_u:rp_x , rank_u:rp_y].T
        dim_Ra = rp_x - rank_u

    # print(P)
    # print(Ra)
    
    v = rank_u + a_y
    M = rp[0:rank_u, rank_u:v]

    ub = np.concatenate((V, B), axis=1)
    qq, rq = np.linalg.qr(ub, mode='complete')
    qq_x,qq_y = qq.shape
    rq_x, rq_y = rq.shape

    # print(qq)
    # print(rq)


    if qq_y == rank_u:
        Q = qq[:,rank_u : qq_y]
        Rb = rq[rank_u:rq_x , rank_u:rq_y]
        dim_Rb = 0
    else:
        Q = qq[:,rank_u : qq_y]
        Rb = rq[rank_u:rq_x , rank_u:rq_y]
        dim_Rb = rq_x - rank_u

    # print(Q)
    # print(Rb)

    N = V.T @ B
    # print(N)

    ##########################################
    ####### Creating Augmented Sigma
    ##########################################
    # Ra_flatten = Ra.reshape(np.product(Ra.shape),1)
    Ra_flatten = Ra.T
    temp_0 = np.concatenate((M, Ra_flatten), axis=0)
    # print(temp_0)

    if Rb.shape[1] == 0:
        temp_1 = N
    else:
        temp_1 = np.concatenate((N, Rb), axis=0)
    # print(temp_1)
    
    # print('\ntemp0 shape ', temp_0.shape)
    # print('\ntemp1 shape ', temp_1.shape)
    S_augmented = temp_0 @ temp_1.T
    # print(S_augmented)

    S_new = np.zeros((rank_u + dim_Ra, rank_u + dim_Rb))
    S_new[0:S.shape[0], 0: S.shape[1]] = S
    S_new = S_new + S_augmented
    # print(S_new)

    u, sig, vh = np.linalg.svd(S_new, full_matrices=False)

    U_new = np.concatenate((U,P),axis=1) @ u
    S_new = np.diag(sig)

    V_new = np.concatenate((V,Q),axis=1) @ vh.T

    return U_new, S_new, V_new

def update_svd(m_old, m_new):
    if m_old.shape[1] != m_new.shape[1]:
        print('\nNew matrix muss has the same colums as old matrix!')
        return
    m,n = m_old.shape
    c = m_new.shape[0] - m
    u, s, v = np.linalg.svd(m_old, full_matrices=False)

    B = m_new[m:, :].T
    A = np.concatenate((np.zeros((m, c)),np.eye(c)), axis=0)
    S = np.diag(s)
    U = np.concatenate((u,np.zeros((c,n))),axis=0)
    V = v.T 
    u_new, s_new, v_new = increment_svd(U, S, V , A, B)

    e = np.linalg.norm(m_new - np.dot(u_new , np.dot( s_new, v_new.T)),2)
    print('Error is', e)
    



if __name__ == '__main__':
    m = 17
    n = 10
    X = np.random.uniform(0,5, (m,n))
    # X = np.array(range(28)).reshape((n,m)).T
    # print(X)
    # u, s, v = np.linalg.svd(X, full_matrices=False)
    # print(u)

    # new_row = np.random.uniform(0,2,(n,1))
    # new_row = np.ones((n,1))
    # new_row = np.ones((n,2))

    # print(new_row)
    # w = m + 1
    # w = m + 2
    # # A = np.zeros((w,1))
    # B = new_row
    # # A[w-1,0] = 1
    # A = np.concatenate((np.zeros((m, 2)),np.eye(2)), axis=0)
    # # print(A)
    # s = np.diag(s)
    # print(s)
    # temp = np.zeros((1,n))
    # temp = np.zeros((2,n))

    # U_0 = np.concatenate((u,temp),axis=0)
    # X_0 = np.concatenate((X,temp), axis=0)
    # print('\nU_0',U_0)

    # u_new, s_new, v_new = increment_svd(U_0, s, v.T , A, B)
    new_data = np.random.uniform(0,4,(30,n))
    m_new = np.concatenate((X,new_data), axis=0)
    update_svd(X, m_new)
    # t = u_new @ s_new
    # print(t @ v_new.T)
    # e = np.linalg.norm((X_0 + A @ B.T) - np.dot(u_new , np.dot( s_new, v_new.T)),2)
    # print('Error is', e)













