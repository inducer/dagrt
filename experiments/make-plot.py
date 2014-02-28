#! /usr/bin/env/python
import sqlite3 as sqlite
import numpy as np
import numpy.linalg as la  # noqa
from plot_tools import auto_xy_reshape, unwrap_list


import traits.api as trt
from mayavi.tools.mlab_scene_model import \
        MlabSceneModel
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.core.ui.mayavi_scene import MayaviScene
import traitsui.api as tui
print "done importing"


def main():
    import sys
    db_conn = sqlite.connect(sys.argv[1], timeout=30)

    all_angles = unwrap_list(
            db_conn.execute(
                "select distinct angle from data where angle is not null"))
    print "angles:", all_angles
    all_methods = unwrap_list(
            db_conn.execute(
                "select distinct method from data where method is not null"))
    print "methods:", all_methods
    all_mat_types = unwrap_list(
            db_conn.execute(
                "select distinct mat_type from data where mat_type is not null"))
    print "mat_types:", all_mat_types
    all_mat_types[0:2] = all_mat_types[1::-1]
    all_substep_counts = unwrap_list(
            db_conn.execute(
                "select distinct substep_count from data "
                "where substep_count is not null"))
    print "substep_counts:", all_substep_counts
    #all_stable_steps = unwrap_list(
            #db_conn.execute("select distinct stable_steps from data"))

    class Visualization(trt.HasTraits):
        scene = trt.Instance(MlabSceneModel, ())
        angle = trt.Range(0, len(all_angles)-1)
        method = trt.Range(0, len(all_methods)-1)
        mat_type = trt.Range(0, len(all_mat_types)-1)
        substep_count = trt.Range(0, len(all_substep_counts)-1)
        #stable_steps = Range(0, len(all_stable_steps)-1)

        def __init__(self):
            trt.HasTraits.__init__(self)

            x, y, z = self.get_data(
                    mat_type=all_mat_types[0],
                    substep_count=all_substep_counts[0],
                    method=all_methods[0],
                    angle=all_angles[0],
                    #stable_steps=all_stable_steps[0],
                    )

            self.plot = self.scene.mlab.mesh(x, y, z)
            self.first = True
            self.axes = None

        def get_data(self, mat_type, substep_count, method, angle):
            qry = db_conn.execute(
                    "select ratio, offset, dt from data"
                    " where method=? and angle=?"
                    " and mat_type=? and substep_count=?"
                    #" and stable_steps=?"
                    " and offset <= ?+1e-10"
                    " order by ratio, offset",
                    (method, angle, mat_type, substep_count, np.pi))
            x, y, z = auto_xy_reshape(qry)

            import mrab_stability
            factory = getattr(mrab_stability, mat_type)
            print "------------------------------"
            print mat_type, method, substep_count, angle/np.pi
            if x:
                ratio = x[0]
                print "matrices for ratio=%g" % ratio

                offset_step = max(len(y)//20, 1)
                for offset in y[::offset_step]:
                    print repr(factory(
                            ratio=ratio,
                            angle=angle,
                            offset=offset)()), offset/np.pi
            else:
                print "EMPTY"

            x = np.array(x)
            y = np.array(y)
            xnew = np.cos(y)*x[:, np.newaxis]
            ynew = np.sin(y)*x[:, np.newaxis]

            return xnew, ynew, z

        @trt.on_trait_change('angle,method,mat_type,substep_count')
        def update_plot(self):

            mat_type = all_mat_types[int(self.mat_type)]
            substep_count = all_substep_counts[int(self.substep_count)]
            method = all_methods[int(self.method)]
            angle = all_angles[int(self.angle)]
            #stable_steps = all_stable_steps[int(self.stable_steps)]

            x, y, z = self.get_data(
                    mat_type=mat_type,
                    substep_count=substep_count,
                    method=method,
                    angle=angle)

            self.plot.mlab_source.set(x=x, y=y, scalars=z, z=z)

            if self.first:
                self.axes = self.scene.mlab.axes()
                self.axes.axes.use_data_bounds = True
                self.scene.mlab.scalarbar(
                        orientation="vertical", title="stable dt")
                self.first = False

        # layout of the dialog
        view = tui.View(
                tui.Item('scene',
                    editor=SceneEditor(scene_class=MayaviScene),
                    show_label=False),
                tui.VGroup('_', 'angle', 'method', 'mat_type', 'substep_count'),
                width=1024, height=768, resizable=True)

    visualization = Visualization()
    visualization.configure_traits()


if __name__ == "__main__":
    main()
