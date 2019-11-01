from dolfin import *
import function
import hdw
import bdry_2 as bdry

def rk(var_list, temp_list, rhs_list, form_packs, func_packs, variational_forms, dt):
    hdw.project_functions(var_list, temp_list)
    for idx in range(len(form_packs)):
        hdw.project_functions(form_packs[idx], func_packs[idx])
    
    for idx in range(len(variational_forms)):
        a, L = lhs(variational_forms[idx]), rhs(variational_forms[idx])
        solve(a == L, rhs_list[idx])

    for idx in range(len(var_list)):
        var_list[idx].assign(var_list[idx] + rhs_list[idx]*dt)
        #bdry.apply_bdry_conditions(var_list, rhs_list, dt)

    
def rk2(var_list, temp_list, rhs_list, form_packs, func_packs, variational_forms, dt):
    hdw.project_functions(var_list, temp_list)
    for stage in range(2): 
        for idx in range(len(form_packs)):
            hdw.project_functions(form_packs[idx], func_packs[idx])
        for idx in range(len(variational_forms)):
            a, L = lhs(variational_forms[idx]), rhs(variational_forms[idx])
            solve(a == L, rhs_list[idx])
        for idx in range(len(var_list)):
            var_list[idx].assign(var_list[idx] + rhs_list[idx]*dt)

        bdry.apply_bdry_conditions(var_list, rhs_list, dt)

    final_forms = [0.5*(temp_list[idx] + var_list[idx]) for idx in range(len(var_list))]
    hdw.project_functions(final_forms, var_list) 


def rk3(var_list, temp_list, rhs_list, form_packs, func_packs, variational_forms, dt):
    hdw.project_functions(var_list, temp_list)

    #compute u1, stored in var_list
    for idx in range(len(form_packs)):
        hdw.project_functions(form_packs[idx], func_packs[idx])
    for idx in range(len(var_list)):
        a, L = lhs(variational_forms[idx]), rhs(variational_forms[idx])
        solve(a == L, rhs_list[idx])
    for idx in range(len(var_list)):
        var_list[idx].assign(var_list[idx] + rhs_list[idx]*dt)
    #bdry.apply_bdry_conditions(var_list, rhs_list, dt)

    #compute u2, stored in var_list
    for idx in range(len(form_packs)):
        hdw.project_functions(form_packs[idx], func_packs[idx])
    for idx in range(len(var_list)):
        a, L = lhs(variational_forms[idx]), rhs(variational_forms[idx])
        solve(a == L, rhs_list[idx])
    for idx in range(len(var_list)):
        var_list[idx].assign(var_list[idx] + rhs_list[idx]*dt)
    #bdry.apply_bdry_conditions(var_list, rhs_list, dt)
    u2_forms = [3.0/4.0*temp_list[idx] + 1.0/4.0*var_list[idx] for idx in range(len(var_list))]
    hdw.project_functions(u2_forms, var_list)

    #compute final u, stored in var_list
    for idx in range(len(form_packs)):
        hdw.project_functions(form_packs[idx], func_packs[idx])
    for idx in range(len(var_list)):
        a, L = lhs(variational_forms[idx]), rhs(variational_forms[idx])
        solve(a == L, rhs_list[idx])
    for idx in range(len(var_list)):
        var_list[idx].assign(var_list[idx] + rhs_list[idx]*dt)
    #bdry.apply_bdry_conditions(var_list, rhs_list, dt)
    final_forms = [1.0/3.0*temp_list[idx] + 2.0/3.0*var_list[idx] for idx in range(len(var_list))]
    hdw.project_functions(final_forms, var_list) 
