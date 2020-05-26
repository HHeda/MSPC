import sqlite3 as s
import numpy as np

class Data(object):
    def __init__(self, filename):
        self.filename = filename
        #self.create(self)
    
    def extract(self, alpha, E, N, phi_ext):
        conn = s.connect(self.filename)
        c = conn.cursor()
        c.execute('''SELECT freqs, dissip, anh, chi, t_w, zpf  FROM data WHERE 
                    alpha = ? and 
                    E = ? and
                    N = ? and 
                    phi_ext = ? ;''', (float(alpha), float(E), int(N), float(phi_ext)))
        freqs, dissip, anh, chi, t_w, zpf = c.fetchone()
        conn.close()
        return (np.frombuffer(freqs, dtype = float).reshape((4,)),
                np.frombuffer(dissip, dtype = float).reshape((4,)),
                np.frombuffer(anh, dtype = float).reshape((4,)),
                np.frombuffer(chi, dtype = float).reshape((4, 4)),
                float(t_w),
                np.frombuffer(zpf, dtype = float).reshape((4,)))
        
    
    def create(self):
        conn = s.connect(self.filename)
        c = conn.cursor()
        c.execute(
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
        c.execute(
            '''CREATE TABLE if not exists keys(key INTEGER UNIQUE)''')
        c.execute('''CREATE UNIQUE INDEX if not exists id ON data(alpha, E, N, phi_ext);''')
        c.execute('''CREATE INDEX if not exists id_key ON keys(key);''')
        conn.commit()
        conn.close()
        
    def save(self, key, alpha, E, N, phi_ext, freqs, dissip, anh, chi, t_w, zpf):
        print('saving...')
        conn = s.connect(self.filename)
        c = conn.cursor()
        c.execute('''SELECT key FROM keys WHERE key = ?''', [key])
        res = c.fetchone() is None
        if res:
            try:
                len(phi_ext)
                test = True
            except:
                test = False

            if not test:
                print('single insertion')
                c.execute('''INSERT OR IGNORE INTO data VALUES (?, ?, ?, ?, ?, ?, ? ,? ,? ,? )''', [float(alpha),
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
                c.executemany('''INSERT OR IGNORE INTO data VALUES (?, ?, ?, ?, ?, ?, ? ,? ,? ,? )''', 
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
            c.execute('''INSERT INTO keys VALUES (?)''', [key])


        conn.commit()
        conn.close()
        print('saved')

        return res
        