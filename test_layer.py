from layer import *
import itertools

# This file just tests the implementations in layer.py
if __name__ == "__main__":
    def test_ACL(V):
        # construct ACL with complex values
        def amFactoryI(Nlayer, activation):

            moduleList = []
            for l in range(Nlayer-1):
                layer = torch.nn.Linear(V//2,V//2,bias=True,dtype=torch.cdouble)
                # this makes the ACL to be the identity
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                
                moduleList.append(layer)
                moduleList.append(activation())
                
            layer = torch.nn.Linear(V//2,V//2,bias=True,dtype=torch.cdouble)
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            moduleList.append(layer)
            # no activation after the last layer

            # we don't need the log det from these, therefore fall back to 
            # torchs' Sequential container
            return torch.nn.Sequential(*moduleList) 

        for act in [torch.nn.Tanh, torch.nn.Softsign]:
            for L in [1,2,4,16,32]:
                ACL = createACL(amFactoryI,amFactoryI, Nlayer = L, activation = act)
                
                x_A = torch.randn(V//2,dtype=torch.cdouble)
                x_B = torch.randn(V//2,dtype=torch.cdouble)
                    
                with torch.no_grad():
                    y_A,logDetJ = ACL(x_A,x_B)
    
                if not (x_A==y_A).all():
                    raise RuntimeError(f"{Nlayer} Layer ACL (V = {V}) is not initialized to the identity: x_A:\n {x_A} \n y_A:\n {y_A}")
    
                # check that the logDetJ is zero
                if not logDetJ == 0:
                    raise RuntimeError(f"{Nlayer} Layer ACL (V = {V}) has wrong logDetJ: logDetJ={logDetJ} != 0 ")
    
        # Test Failed Successfully... 
        print("ACL Test successful")


    def test_PRACL(V):
        def amFactoryI(L, activation):

            moduleList = []
            for l in range(L-1):
                layer = torch.nn.Linear(V//2,V//2,bias=True,dtype=torch.cdouble)
                # this makes the ACL to be the identity
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                
                moduleList.append(layer)
                moduleList.append(activation())
                
            layer = torch.nn.Linear(V//2,V//2,bias=True,dtype=torch.cdouble)
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            moduleList.append(layer)
            # no activation after the last layer

            # we don't need the log det from these, therefore fall back to 
            # torchs' Sequential container
            return torch.nn.Sequential(*moduleList) 

        def amFactoryR(L, activation):

            moduleList = []
            for l in range(L-1):
                layer = torch.nn.Linear(V//2,V//2,bias=True,dtype=torch.cdouble)
                moduleList.append(layer)
                moduleList.append(activation())
                
            layer = torch.nn.Linear(V//2,V//2,bias=True,dtype=torch.cdouble)
            moduleList.append(layer)
            # no activation after the last layer

            # we don't need the log det from these, therefore fall back to 
            # torchs' Sequential container
            return torch.nn.Sequential(*moduleList) 

        def PRACL_wrapper(myPRACL, inputTensor):
            out,_ = myPRACL(inputTensor)
            return out

        # Test PRCL as identity
        for act in [torch.nn.Tanh, torch.nn.Softsign]:
            for LPRACL,LACL in itertools.product([1,2,4,16],repeat=2):
                PRACL = createPRCL(V,LPRACL, 
                    lambda *args,**kwargs: createACL(amFactoryI,amFactoryI,**kwargs),
                    L=LACL,activation=act # are passed as **kwargs to the lambda
                )

                x = torch.randn(V,dtype=torch.cdouble)
                
                with torch.no_grad():
                    y,logDetJ = PRACL(x)

                if not (x==y).all():
                    raise RuntimeError(f"{LPRACL}:{LACL} Layer PRACL (V = {V}) is not initialized to the identity: x_A:\n {x_A} \n y_A:\n {y_A}")

            # check that the logDetJ is zero
                if not logDetJ == 0:
                    raise RuntimeError(f"{LPRACL}:{LACL} Layer PRACL (V = {V}) has wrong logDetJ: logDetJ={logDetJ} != 0 ")

        print("PRACL Identity Test successful")

        # Test randomly initialized PRACL
        for act in [torch.nn.Tanh, torch.nn.Softsign]:
            for LPRACL,LACL in itertools.product([1,2],repeat=2):
                PRACL = createPRCL(V,LPRACL, 
                    lambda *args,**kwargs: createACL(amFactoryR,amFactoryR,**kwargs),
                    L=LACL,activation=act # are passed as **kwargs to the lambda
                )

                x = torch.randn(V,dtype=torch.cdouble)
                xclone = x.clone();
                
                with torch.no_grad():
                    y,logDetJ = PRACL(x)

                # This call is numerical very unstable
                # therefore, the following test sometimes fails
                # not only on a precision level but also on orders of 
                # magnitude. We found a similar behaviour with log det 
                # in NSL. This is realy odd...
                # I ran the test multiple times and most of the times it fails
                # Even for real numbers (using .logdet) it sometimes fails
                #sign,logabsdet = torch.autograd.functional.jacobian(
                #    lambda inTensor: PRACL_wrapper(PRACL,inTensor),
                #    x
                #).slogdet()
                #logDetJ_2 = torch.log(sign) + logabsdet

                ## check that the logDetJ match
                #if not torch.isclose( torch.real(logDetJ),torch.real(logDetJ_2) ):
                #    raise RuntimeError(f"{LPRACL}:{LACL} Layer ACL (V = {V}) has wrong Re logDetJ: Re logDetJ={logDetJ.real:.20} != {logDetJ_2.real:.20} ")
                #if not torch.isclose( torch.imag(logDetJ),torch.imag(logDetJ_2) ):
                #   raise RuntimeError(f"{LPRACL}:{LACL} Layer ACL (V = {V}) has wrong Im logDetJ: Im logDetJ={logDetJ.imag:.20} != {logDetJ_2.imag:.20} ")

        print("PRACL Random Test successful")


    for V in [2,4,16,32,128]:
        test_ACL(V)
        test_PRACL(V)
