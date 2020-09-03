import numpy as np
import sparsesvd
import scipy.sparse as sps

# This function follow the Algorithm of Appendix A of Paper 'Fast online SVD revisions for lightweight recommender systems'
# update the svd from X to X + A @ B.T
# Param is U, S, V of X and A, B
# Return new U, S, V
def increment_svd(U, S, V, A, B):
    # print(U)
    # print(S)
    # print(V)
    # print(A)
    # print(B)
    u_x, u_y = U.shape 
    rank_u = u_y 
    a_x, a_y = A.shape 

    ua = np.concatenate((U, A), axis=1) 
    # print(ua)
    qp,rp = np.linalg.qr(ua, mode='complete') 

    qp_x, qp_y = qp.shape 
    rp_x, rp_y = rp.shape 
    # print(qp)
    # print(rp)
    # print("ua shape {}, qp shape{}, rp shape{}, rank_u {}".format(ua.shape, qp.shape, rp.shape,rank_u))

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
    # M = rp[0:rank_u, rank_u:v]

    ub = np.concatenate((V, B), axis=1)
    # print("\n")
    # print(ub)
    qq, rq = np.linalg.qr(ub, mode='complete')
    qq_x,qq_y = qq.shape
    rq_x, rq_y = rq.shape

    # print(qq)
    # print(rq)
    # print("ub shape {}, qq shape{}, rq shape{}".format(ub.shape, qq.shape, rq.shape))


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
    M = U.T @ A
    N = V.T @ B
    # print(N)

    ##########################################
    ####### Creating Augmented Sigma
    ##########################################
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

    # u, sig, vh = np.linalg.svd(S_new, full_matrices=False)
    temp = sps.csc_matrix(S_new)
    u, sig, vh = sparsesvd.sparsesvd(temp, min(S_new.shape))
    # print("u shape {}".format(u.shape))
    u = u.T


    U_new = np.concatenate((U,P),axis=1) @ u
    S_new = np.diag(sig)

    V_new = np.concatenate((V,Q),axis=1) @ vh.T

    return U_new, S_new, V_new

# This function update the svd when X is appended with more lines
def append_lines_update_svd(m_old, m_new):
    if m_old.shape[1] != m_new.shape[1]:
        print('\nAppend_Lines:New matrix muss has the same colums as old matrix!')
        return
    m,n = m_old.shape
    factor = min(m,n)
    
    # factor = 5
    c = m_new.shape[0] - m
    # print("c {}, factor {}".format(c, factor))
    temp = sps.csc_matrix(m_old)
    u, s, v = sparsesvd.sparsesvd(temp, factor)
  
    u = u.T
    # print("u shape {}".format(u.shape))


    B = m_new[m:, :].T
    A = np.concatenate((np.zeros((m, c)),np.eye(c)), axis=0)
    S = np.diag(s)
    U = np.concatenate((u,np.zeros((c,u.shape[1]))),axis=0)
    V = v.T
    # print("u shape {}, A shape {}, U shape {}, V shape {}".format(u.shape, A.shape, U.shape, V.shape))
    u_new, s_new, v_new = increment_svd(U, S, V , A, B)

    e = np.linalg.norm(m_new - np.dot(u_new , np.dot( s_new, v_new.T)),2)
    print('Error is', e)

# This function update svd when single row or colum in X is updated
def single_line_update_svd(m_old, m_new):
    if m_old.shape != m_new.shape:
        print("\nSingel_line: Matrix shape don't match")
    shape = m_old.shape
    diff = m_new - m_old
    r,c = np.nonzero(diff)
    # print(r,c)
    x = diff[diff != 0]
    # print(x, x.shape)
    if x.shape[0] == shape[0]:
        # update colum
        col = c[0]
        a = x.reshape((shape[0],1))
        b = np.zeros((shape[1],1))
        b[c,0] = 1
    else:
        # update row
        row = r[0]
        a = np.zeros((shape[0],1))
        a[row,0] = 1
        b = x.reshape((shape[1],1)) 
    # print(np.allclose(m_new, m_old + a@b.T))
    temp = sps.csc_matrix(m_old)
    u, s, v = sparsesvd.sparsesvd(temp, min(shape))
    u = u.T 
    v = v.T 
    s = np.diag(s)
    u_new, s_new, v_new = increment_svd(u,s,v,a,b)
    e = np.linalg.norm(m_new - (u_new @ s_new @ v_new.T), 2)
    print("Error is ", e)




if __name__ == '__main__':
    m = 100
    n = 10
    X = np.random.uniform(0,5, (m,n))
   
    new_data = np.random.uniform(0,4,(2,n))
    m_new = np.concatenate((X,new_data), axis=0)
    append_lines_update_svd(X, m_new)
 
    # # update col
    # update = np.random.uniform(0,10,(m,))
    # temp = np.random.randint(0,n-1)
    # X_new = np.copy(X)
    # X_new[:, temp] = update

    # update row
    update = np.random.uniform(0,10,(n,))
    temp = np.random.randint(0,m-1)
    X_new = np.copy(X)
    X_new[temp, :] = update


    single_line_update_svd(X, X_new)

    

   

    











