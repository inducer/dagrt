module sim_types

    type region_type
        integer nGrids
        integer, pointer, dimension(:) :: nCells
        integer, pointer, dimension(:) :: n_grid_dofs
        type(sim_grid_state_type), pointer, dimension(:) :: state
    end type

    type sim_grid_state_type
        real (kind=8), pointer, dimension(:,:) :: conserved_var
        real (kind=8), pointer, dimension(:,:) :: rhs
    end type
end module
