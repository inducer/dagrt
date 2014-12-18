module sim_types

    type region_type
        integer nconserved_vars
        integer state_size
    end type

    type sim_state_type
        real (kind=8), pointer, dimension(:,:) :: conserved_var
        real (kind=8), pointer, dimension(:,:) :: conserved_var_aux
    end type
end module
