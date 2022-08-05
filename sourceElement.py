import numpy as np


class Tri:
    def __init__(self, p0, p1, p2, n):
        self.p = np.zeros((3, 3))
        self.p[:, 0] = p0
        self.p[:, 1] = p1
        self.p[:, 2] = p2
        self.n = n

        self.o, self.x, self.y = self.axisTri()
        self.t = np.zeros((3, 3))

        for i in range(3):
            self.t[:, i] = self.transform(self.p[:, i])

    def axisTri(self):
        o = (self.p[:, 0]+self.p[:, 1]+self.p[:, 2])/3.0
        x = self.p[:, 2]-self.p[:, 0]
        x = x/np.linalg.norm(x)
        y = np.cross(self.n, x)
        return o, x, y

    def transformToTri(self, p):
        r = p-self.o
        t = [np.dot(r, self.x), np.dot(r, self.y), np.dot(r, self.n)]
        return np.array(t)

    def vind(self, p):

        tp = self.transformToTri(p)
        d = np.zeros((3, 3))
        m = np.zeros((3, 3))
        r = np.zeros(3)
        e = np.zeros(3)
        h = np.zeros(3)

        for i in range(3):
            r[i] = np.linalg.norm(tp-t[:, i])
            e[i] = (tp[0]-t[0, i])**2.0 + tp[2]**2.0
            h[i] = (tp[0]-t[0, i])*(tp[1]-t[1, i])
            for j in range(3):
                d[i, j] = np.linalg.norm(self.t[:, j]-self.t[:, i])
                if i == j:
                    m[i, j] = 0.0
                else:
                    m[i, j] = (t[1, j]-t[1, i])/(t[0, j]-t[0, i])

        tu = 0.0
        tv = 0.0
        tw = 0.0
        for i in range(3):
            for j in [1, 2, 0]:
                frac = 1.0/d[i, j]* \
                        np.log((r[i]+r[j]-d[i, j])/(r[i]+r[j]+d[i, j]))
                tu = tu + (t[1, j]-t[1, i])*frac
                tv = tv + (t[0, j]-t[0, i])*frac
                tw = tw + \
                        (np.arctan2(m[i, j]*e[i]-h[i], tp[2]*r[i])- \
                         np.arctan2(m[i, j]*e[j]-h[j], tp[2]*r[j]))

        vel = tu*self.x + tv*self.y + tw*self.n
        vel = vel/(4.0*np.pi)
        return vel
