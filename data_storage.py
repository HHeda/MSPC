import sqlite3 as s
import numpy as np
import pickle


class Data(object):
    def __init__(self, filename):
        self.filename = filename
        self.conn = s.connect(self.filename)
        self.c = self.conn.cursor()

        #self.create(self)
    
    def extract(self, alpha, E, N, phi_ext):
        self.c.execute('''SELECT freqs, dissip, anh, chi, t_w, zpf  FROM data WHERE 
                    alpha = ? and 
                    E = ? and
                    N = ? and 
                    phi_ext = ?;''', (float(alpha), float(E), int(N), float(phi_ext)))
        freqs, dissip, anh, chi, t_w, zpf = self.c.fetchone()
        return (np.frombuffer(freqs, dtype = float).reshape((4,)),
                np.frombuffer(dissip, dtype = float).reshape((4,)),
                np.frombuffer(anh, dtype = float).reshape((4,)),
                np.frombuffer(chi, dtype = float).reshape((4, 4)),
                float(t_w),
                np.frombuffer(zpf, dtype = float).reshape((4,)))
        
    
    def create(self):
        self.c.execute(
            '''CREATE TABLE if not exists data 
            (alpha REAL,
            E REAL,
            N INTEGER,
            phi_ext REAL, 
            freqs BLOB, 
            dissip BLOB, 
            anh BLOB, 
            chi BLOB, 
            t_w REAL, 
            zpf BLOB)''')
        self.c.execute(
            '''CREATE TABLE if not exists keys(key INTEGER UNIQUE)''')
        self.c.execute(
            '''CREATE TABLE if not exists f_poly(
            alpha REAL,
            E REAL,
            N INTEGER,
            taylor INTEGER, 
            phi_ext REAL, 
            poly_fa BLOB, 
            poly_fb BLOB,
            poly_fp BLOB)''')
        self.c.execute('''CREATE UNIQUE INDEX if not exists id ON data(alpha, E, N, phi_ext);''')
        self.c.execute('''CREATE INDEX if not exists id_key ON keys(key);''')
        self.c.execute('''CREATE UNIQUE INDEX if not exists id_fun ON f_poly(alpha, E, N, taylor, phi_ext);''')

        self.conn.commit()
        self.conn.close()
        self.conn = s.connect(self.filename)
        self.c = self.conn.cursor()
        

        
    def save(self, key, alpha, E, N, phi_ext, freqs, dissip, anh, chi, t_w, zpf):
        print('saving...')
        self.c.execute('''SELECT key FROM keys WHERE key = ?''', [key])
        res = self.c.fetchone() is None
        if res:
            try:
                len(phi_ext)
                test = True
            except:
                test = False

            if not test:
                print('single insertion')
                self.c.execute('''INSERT OR IGNORE INTO data VALUES (?, ?, ?, ?, ?, ?, ? ,? ,? ,? )''', [float(alpha),
                                                                                                    float(E),
                                                                                                    int(N),
                                                                                                    float(phi_ext), 
                                                                                                     freqs.tobytes(), 
                                                                                                     dissip.tobytes(), 
                                                                                                     anh.tobytes(), 
                                                                                                     chi.tobytes(), 
                                                                                                     float(t_w), 
                                                                                                     zpf.tobytes()])
            else:
                print(' multiple insertions')
                freqs = np.swapaxes(freqs, 0, 1)
                dissip = np.swapaxes(dissip, 0, 1)
                anh = np.swapaxes(anh, 0, 1)
                chi = np.swapaxes(np.swapaxes(chi, 1, 2), 0, 1)
                zpf = np.swapaxes(zpf, 0, 1)
                alphas = [alpha for c in phi_ext]
                Es = [E for c in phi_ext]
                Ns = [N for c in phi_ext]
                self.c.executemany('''INSERT OR IGNORE INTO data VALUES (?, ?, ?, ?, ?, ?, ? ,? ,? ,? )''', 
                              [[float(alphas[i]),
                                float(Es[i]),
                                int(Ns[i]),
                                  float(phi_ext[i]), 
                                 freqs[i].tobytes(), 
                                 dissip[i].tobytes(), 
                                 anh[i].tobytes(), 
                                 chi[i].tobytes(), 
                                 float(t_w[i]), 
                                 zpf[i].tobytes()] 
                               for i in range(len(phi_ext))])
            self.c.execute('''INSERT INTO keys VALUES (?)''', [key])


        self.conn.commit()
        self.conn.close()
        self.conn = s.connect(self.filename)
        self.c = self.conn.cursor()
        print('saved')

        return res
        
    def save_polys(self, alpha, E, N, taylor, phi_ext, pa, pb, pp):

        self.c.execute('''INSERT OR IGNORE INTO f_poly VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', [float(alpha),
                                                                                    float(E),
                                                                                    int(N),
                                                                                    int(taylor),
                                                                                    float(phi_ext), 
                                                                                    pickle.dumps(pa),
                                                                                    pickle.dumps(pb),
                                                                                    pickle.dumps(pp)])
        self.conn.commit()
        self.conn.close()
        self.conn = s.connect(self.filename)
        self.c = self.conn.cursor()
        
    def extract_polys(self, alpha, E, N, taylor, phi_ext):
        self.c.execute('''SELECT poly_fa, poly_fb, poly_fp FROM f_poly WHERE 
                    alpha = ? and 
                    E = ? and
                    N = ? and 
                    taylor = ? and
                    phi_ext = ?;''', (float(alpha), float(E), int(N), int(taylor), float(phi_ext)))
        res = self.c.fetchone()
        if res is not None:
            return pickle.loads(res[0]), pickle.loads(res[1]), pickle.loads(res[2])
        else: 
            return None
        
    def reinit_f_poly(self):
        conn = s.connect(self.filename)
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS f_poly")
        conn.commit()
        conn.close()
        self.create()