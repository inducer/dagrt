module sim_types

    type region_type
        integer n_grids
        integer, pointer, dimension(:) :: n_grid_dofs
    end type

    type sim_grid_state_type
        real (kind=8), pointer, dimension(:) :: conserved_var
    end type
end module
