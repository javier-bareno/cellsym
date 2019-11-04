
import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from matplotlib import animation, rc
import matplotlib.ticker as ticker
from scipy.interpolate import RegularGridInterpolator as rgi
from IPython.display import HTML

#from IPython.display import HTML #May beneeded for Jup Notebooks functionality

class Electrode:
    '''A class to handle electrode V-Q characteristic.
    Handles interpolation and generation of a function to get V(q) and q(V)'''

    def __init__(self, el_type = "NCM523", qT = 280, q_min = 0.05, q_max = 230/280):
        self.el_type = el_type
        self.theor_Q = qT #mAh/g
        
        #validity range
        self.V_min = False
        self.V_max = False
        self.q_min =q_min
        self.q_max = q_max
        #interpolated data:
        self.Vgrid =[]
        self.q_grid=[]
        self.num_pts = 100
    
    
    def interp_VQ(self, dset, qT=False, V_label= 'Voltage', q_label='mAh/g', num_pts = 100):
        '''Interpolates exp data to get eos'''
        if qT != False:
            #theoretical cap supplied; otherwise use self.theor_Q
            self.theor_Q = qT
        dset['Qf']=dset[q_label]/self.theor_Q
        self.q_grid=dset['Qf']
        self.V_grid=dset[V_label]
        self.V = rgi((self.q_grid.values,), self.V_grid.values)
        self.q = rgi((self.V_grid.values,), self.q_grid.values)
        return()
        
class SymCell:
    '''A class to track the state of a symmetric cell and perform symple simulations
    The cell contains eos for electrodes
    self.num_cycles number of cycles, starts at zero
    self.cycles, a pd.DataFrame containing the cell state at each step of each cycle
    self.cycle_range list for each cycle [first step, num cycle steps]'''

    def __init__(self, eos, VA = 4, VC=4, loadA=1, loadC=1):
        self.eos = eos #A cellsym.Electrode containing the V-Q of the electrode
        #self.num_cycles = 0
        
        tdic ={
            'loadA':loadA, 'loadC':loadC,
            'VA': VA, 'VC': VC, 'V' : VC - VA,
            'load_f_A': 2 * loadA / (loadA + loadC),
            'load_f_C': 2 * loadC / (loadA + loadC),
            'n2p':loadA/loadC,
            'qA': self.eos.q([VA])[0],
            'qC': self.eos.q([VC])[0], 
            'cyc_type':'Initial state',
            'cyc_subtype': '',
            'cyc_num': 0,
            'Q_cycle':0.0,
            'Q_march':0.0,
            'Q_roll':0.0
        }
        self.cycles = pd.DataFrame(tdic, index=[0])
        self.cycles['Afill'] = 1-self.cycles['qA']
        self.cycles['Cfill'] = 1-self.cycles['qC']
        self.cyc_ranges = [[0, 1]] # first step, num steps
    
    def num_cycles(self):
        # number of cycles beyond initial state, which is both 
        # cycle and step zero
        return(len(self.cyc_ranges) -1)
    
    def cyc_range(self, cycle_num):
        if cycle_num > self.num_cycles():
            return(False)
        else:
            x0 = self.cyc_ranges[cycle_num][0]
            xf = x0 + self.cyc_ranges[cycle_num][1] +1
            return range(xo, xf)
            #  
        
    def Q2V(self, Vcutoff, tol_Q = 1e-4, step_Q = 0.1, qA = False, qC = False):
        '''Returns total charge Q needed to charge cell to Vcutoff voltage
        takes into account electrode imbalance
        Q >0 is charge, Q< 0 is discharge'''
        #print('qA\tVA\tqC\tVC\tV\tretQ')
            
        retQ = 0
        if qA == False:
            qA = self.cycles.iloc[-1]['qA']
        if qC == False:
            qC = self.cycles.iloc[-1]['qC']
        VA = self.eos.V([qA])[0]
        VC = self.eos.V([qC])[0]
        V=VC-VA
        #determine search direction:
        dV = Vcutoff-V #needed change
        CorD = np.sign(dV)
        #search at this level:
        while np.sign(Vcutoff-V) == CorD:
            #print('{:5.4f}\t{:5.2f}\t{:5.4f}\t{:5.2f}\t{:5.3f}\t{:5.4f}'.format(qA,VA, qC, VC, V, retQ))
            qC += step_Q * CorD / self.cycles.iloc[-1]['load_f_C']
            qA -= step_Q * CorD / self.cycles.iloc[-1]['load_f_A']
            VC = self.eos.V([qC])[0]
            VA = self.eos.V([qA])[0]
            V=VC-VA
            retQ += step_Q * CorD
        if (step_Q <= tol_Q) or (abs(Vcutoff-V)<1e-6):
            #print('tol reached: ')
            #print(retQ)
            return(retQ)
        else:
            #print('keep searching: ', step_Q)
            rQ = retQ + self.Q2V(Vcutoff, tol_Q = tol_Q,
             step_Q = 0.1 * step_Q, qA = qA, qC = qC)
            #print(rQ)
            return(rQ)

    def Q2V_redox(self, Vcutoff, qCox=0., qAox=0., tol_Q = 1e-4, 
    step_Q = 0.1, qA = False, qC = False, debug = False):
        '''Returns total charge Q needed to charge cell to Vcutoff voltage,
        in the presence of redox currents at electrodes
        takes into account electrode imbalance
        Q >0 is charge, Q< 0 is discharge'''
        #print('qA\tVA\tqC\tVC\tV\tretQ')
            
        retQ = 0
        if qA == False:
            qA = self.cycles.iloc[-1]['qA']
        if qC == False:
            qC = self.cycles.iloc[-1]['qC']
        VA = self.eos.V([qA])[0]
        VC = self.eos.V([qC])[0]
        V=VC-VA
        #determine search direction:
        dV = Vcutoff-V #needed change
        CorD = np.sign(dV)
        #search at this level:
        while np.sign(Vcutoff-V) == CorD:
            #print('{:5.4f}\t{:5.2f}\t{:5.4f}\t{:5.2f}\t{:5.3f}\t{:5.4f}'.format(qA,VA, qC, VC, V, retQ))
            

            dqC = (1-CorD*qCox) * step_Q * CorD / self.cycles.iloc[-1]['load_f_C']
            qC += dqC
            dqA = (1+CorD*qAox) * step_Q * CorD / self.cycles.iloc[-1]['load_f_A']
            qA -= dqA
            VC = self.eos.V([qC])[0]
            VA = self.eos.V([qA])[0]
            V=VC-VA
            retQ += step_Q * CorD
            if type(debug) == type(pd.DataFrame()):
                s=pd.Series({
                    'retQ':retQ, 'qA':qA, 'qC':qC,
                    'VA':VA, 'VC':VC, 'dqA' : dqA,
                    'dqC':dqC, 'dQ':step_Q
                })
                debug = debug.append(s, ignore_index=True)
            
            
        if (step_Q <= tol_Q) or (abs(Vcutoff-V)<1e-6):
            #print('tol reached: ')
            #print(retQ)
            return(retQ, debug)
        else:
            #print('keep searching: ', step_Q)
            drQ =self.Q2V_redox(
                Vcutoff, tol_Q = tol_Q, step_Q = 0.1 * step_Q, qA = qA+dqA , qC = qC-dqC,
                 qCox=qCox, qAox=qAox, debug = debug)
            rQ = retQ - CorD * step_Q + drQ[0]
            debug = drQ[-1]
            #print(rQ)
            return(rQ, debug)  
        
    def charge2V(self, Vcutoff, step_Q = 5e-3, num_steps = False):
        '''Charge to Vcutoff 
        if num_steps = False: equal charge increments close to step_Q
        else: in num_steps equal charge steps
        Cycle would be charge or discharge depending V-Vcutoff, as needed'''
        #determine charge needed size:
        
        dq = self.Q2V(Vcutoff)
        if num_steps != False:
            q_step = dq/num_steps
        else:
            num_steps = abs(int(round(dq/step_Q)))
            q_step = dq/num_steps
        
        CorD = np.sign(dq)

        #record cycle as pandas dataframe
        init_state = self.cycles.iloc[-1]
        
        #new_data.iloc[0] is starting point of cycle. Do not copy
        # to self.cycles

        new_data = pd.DataFrame({
            'Q_cycle': np.linspace(0., dq, num_steps+1),
            'loadA': init_state['loadA'] * np.ones(num_steps+1),
            'loadC': init_state['loadC'] * np.ones(num_steps+1),
            #load_f_A/C control how external charge dq affects
            #electrode SOC. Determine wrt initial loading
            'load_f_A': init_state['load_f_A']* np.ones(num_steps+1),
            'load_f_C': init_state['load_f_C']* np.ones(num_steps+1),
            'n2p': init_state['n2p'] * np.ones(num_steps+1)
        })
        qA0 = self.cycles.iloc[-1]['qA']
        qAf = qA0 - dq/self.cycles.iloc[-1]['load_f_A']
        new_data['qA']=np.linspace(qA0, qAf, num_steps+1)
        new_data['VA']=self.eos.V(new_data['qA'])

        qC0 = self.cycles.iloc[-1]['qC']
        qCf = qC0 + dq/self.cycles.iloc[-1]['load_f_C']
        new_data['qC']=np.linspace(qC0, qCf, num_steps+1)
        new_data['VC']=self.eos.V(new_data['qC'])

        new_data['V']= new_data['VC'] - new_data['VA']
        new_data['Afill']=1 -new_data['qA']
        new_data['Cfill']=1 -new_data['qC']
        new_data['cyc_type'] = 'C' if CorD > 0 else 'D'
        new_data['cyc_subtype'] = 0
        new_data['cyc_num'] = len(self.cyc_ranges)
        new_data['Q_march'] = init_state['Q_march'] + new_data['Q_cycle']
        new_data['Q_roll'] =  init_state['Q_roll'] + abs(new_data['Q_cycle'])
        
        ndf = pd.DataFrame(new_data).iloc[1:]
        self.cyc_ranges.append([len(self.cyc_ranges), len(ndf) ])
        self.cycles = self.cycles.append(ndf, ignore_index = True, sort=False)
        return(self)

    def looseAM(self, Aloss = 0, Closs =0, absolute = False, num_steps = 1):
        '''A cycle of active material loss, keeping the SOC of electrodes.
        A negative loss is a gain. 
        The loss is relative wrt initial electrode loadings (mult 1-loss)
        If absolute, then absolute loss (subtract loss)'''
        if Aloss == 0:
            if Closs != 0:
                ctype = 'C'
            else:
                return(self)
        else:
            if Closs != 0:
                ctype = 'AC'
            else:
                ctype = 'A'
        
        init_state = self.cycles.iloc[-1]
        
        
        
        loadA0 = init_state['loadA']
        loadAf = loadA0 - Aloss if absolute else loadA0 * (1-Aloss)
        loadC0 = init_state['loadC']
        loadCf = loadC0 - Closs if absolute else loadC0 * (1-Closs)
        new_data = pd.DataFrame({
            'loadA': np.linspace(loadA0, loadAf, num_steps+1),
            'loadC': np.linspace(loadC0, loadCf, num_steps+1)
        }).iloc[1:]
        new_data['VA'] = init_state['VA'] 
        new_data['VC'] = init_state['VC'] 
        new_data['V'] = new_data['VC'] - new_data['VA']

        new_data['qA'] = init_state['qA'] 
        new_data['qC'] = init_state['qC'] 
        new_data['Q_cycle'] = 0.
        new_data['Q_march'] = init_state['Q_march'] + new_data['Q_cycle']
        new_data['Q_roll'] =  init_state['Q_roll'] + abs(new_data['Q_cycle'])
        
        new_data['cyc_type'] = 'EL'
        new_data['cyc_subtype'] = ctype
        load_f_norm = self.cycles.iloc[0]['loadA']
        #print(type(load_f_norm))
        load_f_norm += self.cycles.iloc[0]['loadC']
        new_data['load_f_A'] = (2 / load_f_norm) * new_data['loadA'] 
        #new_data['load_f_A'] /= new_data['loadA'] + new_data['loadC']
        new_data['load_f_C'] = (2 / load_f_norm) * new_data['loadC']
        #new_data['load_f_C'] /= new_data['loadA'] + new_data['loadC']    
        new_data['n2p'] = new_data['loadA'] / new_data['loadC']
        new_data['Afill'] = 1-new_data['qA']
        new_data['Cfill'] = 1-new_data['qC']

        new_data['cyc_num'] = len(self.cyc_ranges)
        
        self.cyc_ranges.append([len(self.cyc_ranges), len(new_data) ])
        self.cycles = self.cycles.append(new_data, ignore_index = True, sort=False)
        return(self)

    def rearrangeLi(self, Aloss = 0, Closs =0, absolute = False, num_steps = 1):
        ''' Phase transform AM, causing AM loss but keep total Li in remaining 
        active AM
        
        The loss is relative wrt current electrode loadings (mult 1-loss)
        If absolute, then absolute loss (subtract loss)'''
        
        if Aloss == 0:
            if Closs != 0:
                ctype = 'C'
            else:
                return(self)
        else:
            if Closs != 0:
                ctype = 'AC'
            else:
                ctype = 'A'
        
        init_state = self.cycles.iloc[-1]
        
        loadA0 = init_state['loadA']
        loadAf = loadA0 - Aloss if absolute else loadA0 * (1-Aloss)
        loadC0 = init_state['loadC']
        loadCf = loadC0 - Closs if absolute else loadC0 * (1-Closs)
        new_data = pd.DataFrame({
            'loadA': np.linspace(loadA0, loadAf, num_steps+1),
            'loadC': np.linspace(loadC0, loadCf, num_steps+1)
        }).iloc[1:]
        #loadA0 * qA0 = loadAF * qAF
        new_data['Afill'] = init_state['Afill'] * loadA0 / new_data['loadA']
        new_data['Cfill'] = init_state['Cfill'] * loadC0 / new_data['loadC']
        new_data['qA'] = 1-new_data['Afill']
        new_data['qC'] = 1-new_data['Cfill']
        
        new_data['VA'] = self.eos.V(new_data['qA'])
        new_data['VC'] = self.eos.V(new_data['qC']) 
        new_data['V'] = new_data['VC'] - new_data['VA']

        new_data['Q_cycle'] = 0.
        new_data['Q_march'] = init_state['Q_march'] + new_data['Q_cycle']
        new_data['Q_roll'] =  init_state['Q_roll'] + abs(new_data['Q_cycle'])
        
        new_data['cyc_type'] = 'EL'
        new_data['cyc_subtype'] = ctype
        load_f_norm = self.cycles.iloc[0]['loadA']
        #print(type(load_f_norm))
        load_f_norm += self.cycles.iloc[0]['loadC']
        new_data['load_f_A'] = (2 / load_f_norm) * new_data['loadA'] 
        #new_data['load_f_A'] /= new_data['loadA'] + new_data['loadC']
        new_data['load_f_C'] = (2 / load_f_norm) * new_data['loadC']
        #new_data['load_f_C'] /= new_data['loadA'] + new_data['loadC']    
        new_data['n2p'] = new_data['loadA'] / new_data['loadC']
        

        new_data['cyc_num'] = len(self.cyc_ranges)
        
        self.cyc_ranges.append([len(self.cyc_ranges), len(new_data) ])
        self.cycles = self.cycles.append(new_data, ignore_index = True, sort=False)
        return(self)

    def redox_charge2V(self, Vcutoff, qCox=0. , qAox=0. , step_Q = 5e-3, num_steps = False):
        '''Charge to Vcutoff, considering redox currents at each electrode.
         Sign convention: Charge and oxidation currents are possitive 
        Redox currents are expressed as fraction of applied current (q/step)
        if num_steps = False: equal charge increments close to step_Q
        
        else: in num_steps equal charge steps
        Cycle would be charge or discharge depending V-Vcutoff, as needed'''
        #determine charge needed size:
        dq = self.Q2V_redox(Vcutoff, qCox=qCox , qAox=qAox)[0]
        if num_steps != False:
            q_step = dq/num_steps
        else:
            num_steps = abs(int(round(dq/step_Q)))
            q_step = dq/num_steps
        
        CorD = np.sign(dq)

        #record cycle as pandas dataframe
        init_state = self.cycles.iloc[-1]
        
        #new_data.iloc[0] is starting point of cycle. Do not copy
        # to self.cycles

        new_data = pd.DataFrame({
            'Q_cycle': np.linspace(0., dq, num_steps+1),
            'loadA': init_state['loadA'] * np.ones(num_steps+1),
            'loadC': init_state['loadC'] * np.ones(num_steps+1),
            #load_f_A/C control how external charge dq affects
            #electrode SOC. Determine wrt initial loading
            'load_f_A': init_state['load_f_A']* np.ones(num_steps+1),
            'load_f_C': init_state['load_f_C']* np.ones(num_steps+1),
            'n2p': init_state['n2p'] * np.ones(num_steps+1)
        })

        qA0 = self.cycles.iloc[-1]['qA']
        dqA = (1+CorD*qAox) * dq / self.cycles.iloc[-1]['load_f_A']
        qAf = qA0 - dqA
        new_data['qA']=np.linspace(qA0, qAf, num_steps+1)
        new_data['VA']=self.eos.V(new_data['qA'])

        qC0 = self.cycles.iloc[-1]['qC']
        dqC = (1-CorD*qCox) * dq / self.cycles.iloc[-1]['load_f_C']
        qCf = qC0 + dqC
        new_data['qC']=np.linspace(qC0, qCf, num_steps+1)
        new_data['VC']=self.eos.V(new_data['qC'])

        new_data['V']= new_data['VC'] - new_data['VA']
        new_data['Afill']=1 -new_data['qA']
        new_data['Cfill']=1 -new_data['qC']
        new_data['cyc_type'] = 'C' if CorD > 0 else 'D'
        new_data['cyc_subtype'] = 'redox'
        new_data['cyc_num'] = len(self.cyc_ranges)
        new_data['Q_march'] = init_state['Q_march'] + new_data['Q_cycle']
        new_data['Q_roll'] =  init_state['Q_roll'] + abs(new_data['Q_cycle'])
        
        ndf = pd.DataFrame(new_data).iloc[1:]
        self.cyc_ranges.append([len(self.cyc_ranges), len(ndf) ])
        self.cycles = self.cycles.append(ndf, ignore_index = True, sort=False)
        return(self)

class SymCellDashboard:
    '''A class to handle ploting cell data and cycle animations'''

    def __init__(self, cell):
        self.cell = cell #A cellsym SymCell object containing data
        self.fig = False
        self.ax =False
        self.fig_size = (7.12, 4.76)

    def init_fig(self, watermark = False, show=False):
        '''Initialize figure plot'''
        self.fig, self.ax = plt.subplots(2,2, figsize= self.fig_size)
        
        #ax[0][0] is V-Q of electrodes
        tq = np.linspace(self.cell.eos.q_min, self.cell.eos.q_max, 25 )
        tV = self.cell.eos.V(tq)
        self.VQ = self.ax[0][0].plot(tq,tV)
        self.ax[0][0].set_xlim(-0.05, 1.05)
        #ax[0][0].set_ylim(3, 5)
        self.ax[0][0].set_xlabel('Q (fraction)')
        self.ax[0][0].set_ylabel('Volatge (V)')
        self.ax[0][0].set_title(self.cell.eos.el_type + ' v. Li')
        self.VAmark = self.ax[0][0].plot([self.cell.qA ], [self.cell.VA], 'ro')[0]
        self.VCmark = self.ax[0][0].plot([self.cell.qC], [self.cell.VC], 'go')[0]

        #ax[0][1] is V-Q of cell
        self.ax[0][1].set_xlim(-0.05, 0.4)
        self.ax[0][1].set_ylim(0, 0.75)
        self.ax[0][1].set_xlabel('Q (fraction)')
        self.ax[0][1].set_ylabel('Volatge (V)')
        self.ax[0][1].set_title(self.cell.eos.el_type + ' symmetric cell')
        self.OCV_plot = self.ax[0][1].plot([], [], 'b')[0]
        self.OCV_mark = self.ax[0][1].plot([], [], 'bo')[0]
        self.OCV_x, self.OCV_y = [], []

        #ax[1][0] is electrode filing diagram
        self.ax[1][0].set_xlim(1.05, -.05)
        self.ax[1][0].set_ylim(0, 24)
        self.ax[1][0].set_yticks([6,18])
        self.ax[1][0].set_yticklabels(['Neg', 'Pos'])
        #ax[1][0].yaxis.set_visible(False)
        self.ax[1][0].set_xlabel('Q (fraction)')
        #ax[1][0].set_ylabel('Volatge (V)')
        self.ax[1][0].set_title('ELectrode filling by Li')
        tx = np.array([0,1,1,0])
        tfA = 1-self.cell.qA
        tyA = [1,1, 11, 11]
        tfC = 1-self.cell.qC
        tyC = [13, 13, 23, 23]
        self.ax[1][0].fill(tx, tyA, edgecolor = 'k', linewidth = 2, facecolor='gray')
        self.ax[1][0].fill(tx, tyC, edgecolor = 'k', linewidth = 2, facecolor='gray')
        self.Afill=self.ax[1][0].fill((1-tfA)*tx, tyA, 'r', edgecolor = 'k', linewidth = 2)[0]
        self.Cfill=self.ax[1][0].fill((1-tfC)*tx, tyC, 'g', edgecolor = 'k', linewidth = 2)[0]

         #ax[1][0] is cycle count properties
         #assumes cell starts discharged
        self.ax[1][1].set_xlim(0,10)
        self.ax[1][1].set_ylim(3.5, 4.75)
        self.ax[1][1].set_xlabel('Half-cycle count')
        self.ax[1][1].set_ylabel('Electrode V')
        # self.Electrode_V0 =[[0], [4.0]]
        # Electrode_VC =[[], []]
        # Electrode_VA =[[], []]
        self.Electrode_V_n = self.ax[1][1].plot([], [], 'b--o', [], [],'g--o', [], [], 'r--o')
        self.cellQaxis = self.ax[1][1].twinx()
        self.cellQaxis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        self.cellQaxis.set_ylim(0,0.35)
        self.cellQaxis.set_ylabel('* Symmetric cell Q', color='m')
        self.cellQaxis.tick_params('y',colors='m' )
        #self..cellQ=[[0],[0]]
        self.cell_Q_n=self.cellQaxis.plot([], [], 'm--*')[0]
        #ax[1][1].set_title('')

        if watermark:
            fig.text(0.5, 0.5, "Javier Bareño", size=95, rotation=30.,
                    ha="center", va="center", alpha = 0.25,
                    )

        plt.tight_layout()
        if show:
            plt.show()
        return(self)
    
    def clear(self):
        self.fig = False
        self.ax = False
        return(self)
    
    def show(self):
        plt.show()
        return(self)

    def plot_cycles(self, watermark = False, show=False, plot_some = False, style = 'rolling', Vlimits = False):
        '''Plots (doesnt animate) the cycles
        If not plot_some, plot subset (first, last)
        style controls how the cycles are ploted v. Q (proxy for time):
        'rolling': one after another, abs(Q)
        'march' : start at end, come back
        'reset': abs(Q) start each cycle at zero
        '''
        if self.fig == False:
            self.init_fig(watermark)
        if plot_some == False:
            plot_some = range(len(self.cell.cycles))
        
        tVA = self.cell.cycles[plot_some[-1]][0][8]
        tVC = self.cell.cycles[plot_some[-1]][0][10]
        tqA = self.cell.cycles[plot_some[-1]][0][7]
        tqC = self.cell.cycles[plot_some[-1]][0][9]

        self.VAmark.set_data(tqA, tVA)
        self.VCmark.set_data(tqC, tVC)

        txA= [1-tqA, 1-tqA, 0, 0]
        tyA= [1, 11, 11, 1]
        self.Afill.set_xy( np.array([[txA[i], tyA[i]] for i in range(len(txA))]) )
        txC=[1-tqC, 1-tqC, 0, 0]
        tyC = np.array([0,10,10,0])/self.cell.n2p
        tyC = 23* np.ones(4) - tyC
        self.Cfill.set_xy( np.array([[txC[i], tyC[i]] for i in range(len(txC))]) )

        if style == 'rolling':
            Q,VA,VC = [],[],[]
            Q0=0
            for i in plot_some:
                QQ = abs(self.cell.cycles[i][1]['Q'].values)
                QQ += Q0 * np.ones(len(QQ))
                Q0=QQ[-1]
                Q.extend(QQ)
                VA.extend(self.cell.cycles[i][1]['VA'].values)
                VC.extend(self.cell.cycles[i][1]['VC'].values)
                self.ax[0][1].set_xlim(min(Q)-0.05, max(Q)+0.05)
                self.OCV_plot.set_data(Q, np.array(VC)-np.array(VA))
        elif style == 'march':
            Q,VA,VC = [],[],[]
            Q0=0
            for i in plot_some:
                QQ = self.cell.cycles[i][1]['Q'].values
                QQ += Q0 * np.ones(len(QQ))
                Q0=QQ[-1]
                Q.extend(QQ)
                VA.extend(self.cell.cycles[i][1]['VA'].values)
                VC.extend(self.cell.cycles[i][1]['VC'].values)
                self.ax[0][1].set_xlim(min(Q)-0.005, max(Q)+0.005)
                self.OCV_plot.set_data(Q, np.array(VC)-np.array(VA))
        elif style=='reset':
            for i in plot_some:
                QQ = abs(self.cell.cycles[i][1]['Q'].values)
                self.ax[0][1].set_xlim(min(Q)-0.005, max(Q)+0.005)
                self.ax[0][1].plot(QQ, np.array(VC)-np.array(VA), 'b')
            
        
        x0 = plot_some[0]
        xx =[x0]
        xx.extend([x+1 for x in plot_some])
        VA =[self.cell.cycles[x0][0][3]]
        VC =[self.cell.cycles[x0][0][5]]
        Q = [0]
        for i in plot_some:
            VA.append(self.cell.cycles[i][0][8])
            VC.append(self.cell.cycles[i][0][10])
            Q.append(abs(self.cell.cycles[i][0][-1]))
        V =  np.array(VC) -  np.array(VA)
        self.ax[1][1].set_xlim(xx[0]-1, xx[-1]+1)
        self.Electrode_V_n[0].set_data(xx, V)
        self.Electrode_V_n[1].set_data(xx, VC)
        self.Electrode_V_n[2].set_data(xx, VA)

        self.cell_Q_n.set_data(xx[1:], Q[1:])
        self.cellQaxis.set_ylim(0,1.15 * max(Q))

        if Vlimits != False:
            mV, MV = Vlimits
            self.ax[0][1].set_ylim(mV, MV)
            

        if show:
            plt.show()

        return(self)

class SymCellViewDB:
    '''Deprecates SymCellDashboard
    Classto handle plotting cell cycle data in dashboard view.
    Constructs a data set and implements
    still views and animations'''

    def __init__(self, cell, cyc_range=False):
        '''Constructor. cell is SymCell
        cyc_range is iterable with cycles of interest. IF 
        False, all cycles in cell.
        First cycle (index zero) in each of cell.cycles is the ending 
        state of prev. cycle. Need to remove duplicates'''

        
        self.fig=False
        self.ax = False
        if cyc_range==False:
            cyc_range = range(cell.num_cycles()+1)
        self.cyc_range = cyc_range
        self.data=cell.cycles[cell.cycles['cyc_num'].isin(cyc_range)]
        
        self.eos_q = np.linspace(cell.eos.q_min, cell.eos.q_max, 50 )
        self.eos_V = cell.eos.V(self.eos_q)
        self.el_type = cell.eos.el_type
        
        #self.cycle_data contains all cycle steps in cyc_range
        #the index is step number, with step zero initial state at begining of first 
        # cycle in cycle_range

    def plot_DB(
        self, watermark = False, show=False, Vlimits = False,
        width = 7.12, height = 4.75, Q_type = 'roll'
    ):
        '''plots Dash Board view, at end state'''
        self.fig, self.ax = plt.subplots(2,2, figsize= (width, height))
        
        #ax[0][0] is V-Q of electrodes
        self.VQ = self.ax[0][0].plot(self.eos_q,self.eos_V)
        #self.ax[0][0].set_xlim(-0.05, 1.05)
        self.ax[0][0].set_xlabel('Q (fraction)')
        self.ax[0][0].set_ylabel('Volatge (V)')
        self.ax[0][0].set_title(self.el_type + ' v. Li')
        VA = self.data.iloc[-1]['VA']
        qA = self.data.iloc[-1]['qA']
        VC = self.data.iloc[-1]['VC']
        qC = self.data.iloc[-1]['qC']
        self.VA_mark = self.ax[0][0].plot([qA], [VA], 'ro')[0]
        self.VC_mark = self.ax[0][0].plot([qC], [VC], 'go')[0]

        #ax[0][1] is V-Q of cell
        #self.ax[0][1].set_xlim(-0.05, 0.4)
        #self.ax[0][1].set_ylim(0, 0.1)
        self.ax[0][1].set_xlabel('Q (fraction)')
        self.ax[0][1].set_ylabel('Volatge (V)')
        self.ax[0][1].set_title(self.el_type + ' symmetric cell')
        
        self.OCV_mark = self.ax[0][1].plot([], [], 'bo')[0]
        
        if Q_type == 'roll':
            self.OCV_plots = self.ax[0][1].plot(
                self.data['Q_roll'], self.data['V'], 'b')
        elif Q_type == 'march':
            self.OCV_plots = self.ax[0][1].plot(
                self.data['Q_march'], self.data['V'], 'b')
        elif Q_type == 'cycles':
            self.OCV_plots =[]
            for i in self.cyc_range:
                data = self.data[self.data['cyc_num'] == i]
                if data.iloc[0]['cyc_type'] == 'C':
                    x = data['Q_cycle']
                    self.OCV_plots.append(self.ax[0][1].plot(
                    x, data['V'] )[0])
                elif data.iloc[0]['cyc_type'] == 'D':
                    x = abs(data['Q_cycle'])
                    self.OCV_plots.append(self.ax[0][1].plot(
                    x, data['V'] )[0])
        else:
            return(False)
                
        #self.OCV_x, self.OCV_y = [], []

        #ax[1][0] is electrode filing diagram
        self.ax[1][0].set_xlim(1.05, -.05)
        self.ax[1][0].set_ylim(0, 24)
        self.ax[1][0].set_yticks([6,18])
        self.ax[1][0].set_yticklabels(['Neg', 'Pos'])
        #ax[1][0].yaxis.set_visible(False)
        self.ax[1][0].set_xlabel('Q (fraction)')
        #ax[1][0].set_ylabel('Volatge (V)')
        self.ax[1][0].set_title('ELectrode filling by Li')
        tx = np.array([0,1,1,0])
        tfA = self.data.iloc[-1]['Afill'] 
        tyA = [1,1, 11, 11]
        tfC = self.data.iloc[-1]['Cfill']
        tyC = [13, 13, 23, 23]
        self.ax[1][0].fill(tx, tyA, edgecolor = 'k', linewidth = 2, facecolor='gray')
        self.ax[1][0].fill(tx, tyC, edgecolor = 'k', linewidth = 2, facecolor='gray')
        hA = self.data.iloc[-1]['loadA']/self.data.iloc[0]['loadA']
        hC = self.data.iloc[-1]['loadC']/self.data.iloc[0]['loadC']
        tyA = np.array([1,1,1,1]) + hA * np.array([0,0,10,10])
        tyC = 13*np.array([1,1,1,1]) + hC * np.array([0,0,10,10])
        self.Afill=self.ax[1][0].fill((1-tfA)*tx, tyA, 'r', edgecolor = 'k', linewidth = 2)[0]
        self.Cfill=self.ax[1][0].fill((1-tfC)*tx, tyC, 'g', edgecolor = 'k', linewidth = 2)[0]

        #ax[1][0] is cycle count properties
         
        #self.ax[1][1].set_xlim(self.cyc_range[0], self.cyc_range[-1])
        #self.ax[1][1].set_ylim(3.5, 4.75)
        self.ax[1][1].set_xlabel('Half-cycle count')
        self.ax[1][1].set_ylabel('Electrode V')
        # self.Electrode_V0 =[[0], [4.0]]
        # Electrode_VC =[[], []]
        # Electrode_VA =[[], []]
        VA, VC, V, cyc_num, Q_cycle = [],[],[],[], []
        for i in self.cyc_range:
            cyc_num.append(i)
            VA.append(self.data[self.data['cyc_num'] == i].iloc[-1]['VA'])
            VC.append(self.data[self.data['cyc_num'] == i].iloc[-1]['VC'])
            V.append(self.data[self.data['cyc_num'] == i].iloc[-1]['V'])
            Q_cycle.append(abs(self.data[self.data['cyc_num'] == i].iloc[-1]['Q_cycle']))

        self.Electrode_V_n = self.ax[1][1].plot(
            #cyc_num, V, 'b--o', 
            cyc_num, VC,'g--o', cyc_num, VA, 'r--o')
        self.cellQaxis = self.ax[1][1].twinx()
        self.cellQaxis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        self.cellQaxis.set_ylim(0,0.35)
        self.cellQaxis.set_ylabel('* Symmetric cell Q', color='m')
        self.cellQaxis.tick_params('y',colors='m' )
        #self..cellQ=[[0],[0]]
        self.cell_Q_n=self.cellQaxis.plot(cyc_num, Q_cycle, 'm--*')[0]
        #ax[1][1].set_title('')

        if watermark:
            self.fig.text(0.5, 0.5, "© Javier Bareño", size=95, rotation=30.,
                    ha="center", va="center", alpha = 0.25,
                    )
        plt.tight_layout()
        if show:
            plt.show()
        return(self)

    def anim_DB(
        self, watermark = False, show=False, Vlimits = False,
        width = 7.12, height = 4.75, Q_type = 'roll'):
        ''' Animate dashboard '''
        # First, plot DB,  no show
        self.plot_DB( watermark = watermark, show=False, Vlimits = Vlimits,
        width = width, height = height, Q_type = Q_type)
        #delete data from DB plot
        # For EOS plot
        self.VA_mark.set_data([], []) #= self.ax[0][0].plot([qA], [VA], 'ro')[0]
        self.VC_mark.set_data([], []) # = self.ax[0][0].plot([qC], [VC], 'go')[0]
        # For OCV plot
        for i in self.OCV_plots:
            i.set_data([],[])
        self.OCV_plots=[]
        #No need to erase e-fill plot
        #erase cycle plot
        for i in self.Electrode_V_n:
            i.set_data([],[])
        self.cell_Q_n.set_data([],[])
        
        #now populate data
        def aux_anim(i):
            qA = self.data.iloc[i]['qA']
            VA = self.data.iloc[i]['VA']
            self.VA_mark.set_data([qA], [VA])
            return()
        
        animation.FuncAnimation(self.fig, aux_anim, frames=np.arange(0,len(self.data)), \
                                      interval=150, blit=True, repeat=False)
        plt.show()
        return

class BasicView:
    '''Base class to implement cell data viewers.
    Initialization gets subset data of interest, draws canvas
    Implements functions to show static data and animation'''

    def __init__(self, cell, cyc_range=False, anim_int = 40,fig_size = (7.12, 4.76)):
        self.fig_size = fig_size
        #copy data from cell
        if cyc_range==False:
            cyc_range = range(cell.num_cycles()+1)
        self.cyc_range = cyc_range
        self.data=cell.cycles[cell.cycles['cyc_num'].isin(cyc_range)]
        self.len_data = len(self.data)
        self.anim_interval = anim_int #default approx 24 fps
        
        self.eos_q = np.linspace(cell.eos.q_min, cell.eos.q_max, 50 )
        self.eos_V = cell.eos.V(self.eos_q)
        self.el_type = cell.eos.el_type

        #init object functionality
        self.ax_anim = self.init_anim()
        #print('type ax_anim: ', type(self.ax_anim))
        self.fig, self.ax = self.init_canvas()

    def init_canvas(self):
        ''' This function initializes canvas, creating 
        self.ax, self.fig and plotting empty data.
        Returns fig, ax to be stored
        '''
        return(False)
    
    def init_anim(self):
        '''returns tuple of animators to be used by self.anim'''
        return(False, False) 
    
    def anim(self):
        def animate(i):
            ret=[]
            for anim in self.ax_anim:
                ret.extend(anim(i))
            return(ret)
        
        #self.init_canvas()
        myAnimation = animation.FuncAnimation(
            self.fig, animate, frames=np.arange(self.len_data), blit=True, 
            interval = self.anim_interval, repeat=False)
        return(myAnimation)

    def save_anim(self, fname='bogus.mp4', show=False):
        '''Animates and saves to file.
        If show, display in jupyter NB'''
        anim = self.anim()
        anim.save(fname)
        if show:
            video = io.open(fname, 'r+b').read()
            encoded = base64.b64encode(video)
            HTML(data='''<video alt="test" controls>
                            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                        </video>'''.format(encoded.decode('ascii')))
                
    def plot(self, i=-1, fname = '', show=True):
        '''Plots state i (default last). If show, show the plot
        if fname, save to file fname'''
        #essentially, animate one frame only
        #self.init_canvas()
        for anim in self.ax_anim:
            anim(i)

        if(show):
            plt.show()
        if fname != '':
            self.fig.savefig(fname, bbox_inches='tight')
        return(self.fig)
    
    def clear(self):
        '''clears and redraws figure'''
        self.fig.clear()
        self.init_canvas()
        return()

class EOS_view(BasicView):
    '''Test class'''
    def __init__(self, cell, cyc_range=False, anim_int = 40, fig_size = (7.12, 4.76)):
        self.fig_size = fig_size
        BasicView.__init__(self, cell, cyc_range=cyc_range, anim_int=anim_int , fig_size=fig_size )
        
    def init_canvas(self, clear_canvas = True, up_to=False):
        ''' This function initializes canvas, creating 
        self.ax, self.fig and plotting empty data.
        Returns fig, ax to be stored
        '''
        fig, ax = plt.subplots(2,2, figsize=self.fig_size, constrained_layout=True)
        #ax[0][0].lines: [eos VQ, VA marker, VC marker]

        #ax[0][0] is V-Q of electrodes
        #plot eos VQ characteristic
        ax[0][0].plot(self.eos_q,self.eos_V)
        #label axes
        ax[0][0].set_xlabel('Q (fraction)')
        ax[0][0].set_ylabel('Volatge (V)')
        ax[0][0].set_title(self.el_type + ' v. Li')
        #set plot markers for EOS animation
        ax[0][0].plot([], [], 'ro')[0]
        ax[0][0].plot([], [], 'go')[0]

        #ax[1][0] is electrode filing diagram
        ax[1][0].set_xlim(1.05, -.05)
        ax[1][0].set_ylim(0, 24)
        ax[1][0].set_yticks([6,18])
        ax[1][0].set_yticklabels(['Neg', 'Pos'])
        ax[1][0].set_xlabel('[Li] (fraction)')
        ax[1][0].set_title('ELectrode filling by Li')
        tx = np.array([0,1,1,0])
        tyA = [1,1, 11, 11]
        tyC = [13, 13, 23, 23]
        
        ax[1][0].fill(tx, tyA, edgecolor = 'k', linewidth = 2, facecolor='gray')
        ax[1][0].fill(tx, tyC, edgecolor = 'k', linewidth = 2, facecolor='gray')
        # hA = self.data.iloc[-1]['loadA']/self.data.iloc[0]['loadA']
        # hC = self.data.iloc[-1]['loadC']/self.data.iloc[0]['loadC']
        # tyA = np.array([1,1,1,1]) + hA * np.array([0,0,10,10])
        # tyC = 13*np.array([1,1,1,1]) + hC * np.array([0,0,10,10])
        ax[1][0].fill([], [], 'r', edgecolor = 'k', linewidth = 2)[0]
        ax[1][0].fill([], [], 'g', edgecolor = 'k', linewidth = 2)[0]

        # ax[1][1] is cycle count properties
        ax[1][1].set_xlabel('Cycle count')
        ax[1][1].set_ylabel('Electrode V')
        
        self.VA, self.VC, self.V, self.cyc_num, self.Q_cycle = [],[],[],[], []
        self.cyc_ends=[] 
        for i in self.cyc_range:
            self.cyc_num.append(i)
            self.cyc_ends.append(self.data.index[self.data['cyc_num'] == i][-1])
            self.VA.append(self.data[self.data['cyc_num'] == i].iloc[-1]['VA'])
            self.VC.append(self.data[self.data['cyc_num'] == i].iloc[-1]['VC'])
            self.V.append(self.data[self.data['cyc_num'] == i].iloc[-1]['V'])

            self.Q_cycle.append(abs(self.data[self.data['cyc_num'] == i].iloc[-1]['Q_cycle']))
        self.cellQaxis = ax[1][1].twinx()
        self.cellQaxis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        #self.cellQaxis.set_ylim(0,0.35)
        self.cellQaxis.set_ylabel('* Symmetric cell Q', color='m')
        self.cellQaxis.tick_params('y',colors='m' )

        ax[1][1].plot(self.cyc_num, self.VC,'g--o', self.cyc_num, self.VA, 'r--o')
        self.cellQaxis.plot(self.cyc_num, self.Q_cycle, 'm--*')
        
        #self..cellQ=[[0],[0]]
        #self.cellQaxis.plot(self.cyc_num, self.Q_cycle, 'm--*')
        
        if clear_canvas:
            self.cellQaxis.lines[0].set_data([],[])
            for ln in ax[1][1].lines:
                ln.set_data([],[])
        elif up_to != False:
            ax[1][1].lines[0].set_data(
                self.cyc_num[:up_to+1], self.VC[:up_to+1])
            ax[1][1].lines[1].set_data(
                self.cyc_num[:up_to+1], self.VA[:up_to+1])
            self.cellQaxis.lines[0].set_data(
                self.cyc_num[:up_to+1], self.Q_cycle[:up_to+1])

        #plt.tight_layout()
        return(fig, ax)
    
    def init_anim(self):
        '''returns tuple of animators to be used by self.anim'''
        ret = []

        def anim_0(i):
            #animates / plots VQ characteristic on self.axes[0][0]
            VA = self.data.iloc[i]['VA']
            qA = self.data.iloc[i]['qA']
            VC = self.data.iloc[i]['VC']
            qC = self.data.iloc[i]['qC']
            self.ax[0][0].lines[1].set_data([qA], [VA])
            self.ax[0][0].lines[2].set_data([qC], [VC])
            return(self.ax[0][0].lines[1:])
        ret.append(anim_0)

        def anim_1(i):
            tx = np.array([0,1,1,0])
            tfA = self.data.iloc[i]['Afill'] 
            tyA = [1,1, 11, 11]
            tfC = self.data.iloc[i]['Cfill']
            tyC = [13, 13, 23, 23]
            
            hA = self.data.iloc[i]['loadA']/self.data.iloc[0]['loadA']
            hC = self.data.iloc[i]['loadC']/self.data.iloc[0]['loadC']
            tyA = np.array([1,1,1,1]) + hA * np.array([0,0,10,10])
            tyC = 13*np.array([1,1,1,1]) + hC * np.array([0,0,10,10])
            self.ax[1][0].patches[-2].set_xy([[tfA*tx[j], tyA[j]] for j in range(4)])
            self.ax[1][0].patches[-1].set_xy([[tfC*tx[j], tyC[j]] for j in range(4)])
            return(self.ax[1][0].patches[-2:])
        ret.append(anim_1)

        self.j=0
        def anim_2(i):
            
            if i in self.cyc_ends: 
                self.j+=1
                self.cyc_ends.pop(self.cyc_ends.index(i))
                #x=self.cyc_num[:self.j]
                self.cellQaxis.lines[0].set_data(self.cyc_num[:self.j], self.Q_cycle[:self.j])
                self.ax[1][1].lines[0].set_data(self.cyc_num[:self.j], self.VC[:self.j])
                self.ax[1][1].lines[1].set_data(self.cyc_num[:self.j], self.VA[:self.j])
                ret = self.cellQaxis.lines
                ret.extend(self.ax[1][1].lines)
                return(ret)
            else:
                return([])
        ret.append(anim_2)

        return(tuple(ret)) 
  
class DB_ticker(EOS_view):
    '''Dashboard view. x axis of cycle panel is cycle_step (tick number)'''
    def init_canvas(self, clear_canvas = True, up_to=False):
        #initialize canvas and draw three panels
        fig, ax = EOS_view.init_canvas(self, clear_canvas, up_to)
        #initialize ticker
        #ax[0][1] is V-Q of cell.created by base class init_canvas
        ax[0][1].set_xlabel('Step number')
        ax[0][1].set_ylabel('Volatge (V)')
        ax[0][1].set_title(self.el_type + ' symmetric cell')
        # OCV_plot
        ax[0][1].plot(self.data.index, self.data['V'], 'b')
        # OCV_marker
        ax[0][1].plot([], [], 'bo')

        if clear_canvas:
            ax[0][1].lines[0].set_data([],[])
        elif up_to != False:
            #up_to is a cycle number
            data_subset = self.data[self.data['cyc_num'<= up_to]]
            ax[0][1].lines[0].set_data(data_subset.index, data_subset['V'])
            ax[0][1].lines[1].set_data([data_subset.index[-1]], [data_subset['V'][-1]])
        
        return(fig, ax)

    def init_anim(self):
        ret = list(EOS_view.init_anim(self))
        def anim_4(i):
            x = self.ax[0][1].lines[0].get_xdata()
            y = self.ax[0][1].lines[0].get_ydata()
            x.append(self.data.index[i])
            y.append(self.data['V'].iloc[i])
            self.ax[0][1].lines[0].set_data(x,y)
            self.ax[0][1].lines[1].set_data(
                self.data.index[i], self.data['V'].iloc[i]
            )
            return(self.ax[0][1].lines)
        ret.append(anim_4)
        return(tuple(ret))

class static_plot(BasicView):
    '''class to handle static plot views of 
    data in cell'''
    def __init__(self, cell, cyc_range=False, anim_int = 40, fig_size = (7.12, 4.76)):
        self.fig_size = fig_size 
        cyc_0 = min(cell.cycles['cyc_num'])
        cyc_f = max(cell.cycles['cyc_num'])
        cyc_rg = list(range(cyc_0, cyc_f+1))
        ret=[cell.cycles[cell.cycles['cyc_num']== cyc_rg.pop(0)].iloc[-1]]
        for i in cyc_rg:
            #print(i)
            cyc = cell.cycles[cell.cycles['cyc_num']== i].iloc[-1]
            #print(cyc)
            ret.append(cyc)
            #print(ret)
        self.cyc_ends =pd.DataFrame(ret)
        BasicView.__init__(self, cell, cyc_range=cyc_range, anim_int=anim_int , fig_size=fig_size )
        
    def _plot_V_num(self, x, y, ax, title='', style ='b-o'):
        ax.set_xlabel('Cycle number')
        ax.set_ylabel('V (V v. Li)')
        ax.set_title(title)
        ax.plot(x,y,style)
        # dy =max(y)-min(y)
        # ym=min(y)-.05*dy
        # yM = 1.01*max(y)
        # ax.set_ylim(ym, yM)
        return(ax)
    
    def _plot_Q_num(self, xC, yC, xD, yD, ax):
        ax.set_xlabel('Cycle number')
        ax.set_ylabel('Cell charge')
        ax.set_title('Capacity retention')
        ax.plot(xC, yC,'-o', color='saddlebrown', label='Charge')
        ax.plot(xD, yD,'b-o', label='Discharge')
        ax.legend(loc='best')
        # ym=0.99*min([min(yC),min(yD)])
        # yM = 1.01*max([max(yC),max(yD)])
        # ax.set_ylim(ym, yM)
        return(ax)

    def init_canvas(self, kind=0):
        '''function to select kind of plot, called by __init__'''
        if kind==0:
            return(self.init_canvas_0())
        else:
            return(False, False)
    
    def init_canvas_0(self):
        fig,ax=plt.subplots(2,2, constrained_layout=True,  figsize = self.fig_size )
        cc=self.cyc_ends # ref for convenience
        # for axx in ax:
        #     axx.set_xlabel('Cycle number')
        #     axx.set_ylabel('V (V v. Li)')
        y=cc[cc['cyc_type']=='C']['VA']
        x=np.arange(len(y))
        self._plot_V_num(
            x, y, ax[0][0], title='Anode V at top of charge', style ='r-o'
        )

        y=cc[cc['cyc_type']=='C']['VC']
        x=np.arange(len(y))
        self._plot_V_num(
            x, y, ax[0][1], title='Cathode V at top of charge', style ='g-o'
        )
        
        y=cc[cc['cyc_type']=='D']['VC']
        x=np.arange(len(y))
        self._plot_V_num(
            x, y, ax[1][0], title='Electrode V at bottom of discharge', style ='b-o'
        )

        yC=cc[cc['cyc_type']=='C']['Q_cycle'].abs()
        yD=cc[cc['cyc_type']=='D']['Q_cycle'].abs()
        xC=np.arange(len(yC))
        xD=np.arange(len(yD))
        self._plot_Q_num(xC, yC, xD, yD, ax[1][1])
        return(fig, ax)
    
    def plot(self, title ='', fname = '', show=True):
        #assume init_canvas has been called
        if title != '':
            self.fig.suptitle(title)

        if(show):
            plt.show()
        if fname != '':
            self.fig.savefig(fname, bbox_inches='tight')
        return(self.fig)












        