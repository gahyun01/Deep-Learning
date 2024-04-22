import weakref
import numpy as np
import contextlib


# =============================================================================
# Config
# =============================================================================
class Config: # 역전파가 가능한지 여부 확인
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    # 설정값을 변경하기 전에 이전 설정값을 임시로 저장
    old_value = getattr(Config, name)
    # 지정된 속성에 새로운 값 설정
    setattr(Config, name, value)
    try:
        # 설정값을 변경한 후에 코드 블록을 실행
        yield
    finally:
        # 코드 블록 실행 후에 이전 설정값을 복원
        setattr(Config, name, old_value)


# with using_config('enable_backprop', False) 대신
def no_grad():
    return using_config('enable_backprop', False)


# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data # 변수의 데이터를 입력 데이터로 설정
        self.name = name # 변수 이름 저장
        self.grad = None # 미분값 저장
        self.creator = None # 연산을 나타내는 객체
        self.generation = 0 # 세대 수 기록

    @property
    def shape(self): # 다차원 배열의 형상
        return self.data.shape

    @property
    def ndim(self): # 차원 수
        return self.data.ndim

    @property
    def size(self): # 원소 수
        return self.data.size

    @property
    def dtype(self): # 데이터 타입
        return self.data.dtype

    def __len__(self): # 객체 수
        return len(self.data)

    def __repr__(self): # 출력 설정
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        # 세대를 기록함 ( 부모 세대 + 1)
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    # retain_grad=False : 중간  변수의 미분값을모두 None으로 재설정
    def backward(self, retain_grad=False):
        # y.grad = np.array(1.0) 생략을 위한 if문
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()# 함수를 가져온다.
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys) # 함수 f의 역전파 호출 ( 리스트 언팩 )

            # gxs가 튜플이 아니라면 튜플로 변환
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            # 역전파로 전파되는 미분값을 Variable인스턴스 변수 grad에 저장
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            # 중간 미분값 없앰
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref <- 약한 참조


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# 주어진 입력을 NumPy 배열로 변환하는 함수
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 리스트 언팩 ( 원소를 낱개로 풀어서 전달 )
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # enable_backprop = True 일 때만 역전파 코드 실행
        if Config.enable_backprop:
            # 역전파 시 노드에 따라 순서를 정하는데 사용
            self.generation = max([x.generation for x in inputs]) # 세대 설정

            # 각 output Variable 인스턴스의 creator를 현재 Function 객체로 설정
            for output in outputs:
                output.set_creator(self) # 연결 설정
            self.inputs = inputs # 순전파 결과값 기억
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# =============================================================================
# 사칙연산 / 연산자 오버로드
# =============================================================================
class Add(Function): # 더하기
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function): # 곱하기
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function): # 음수
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function): # 빼기
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function): # 나누기
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function): # 거듭제곱
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow