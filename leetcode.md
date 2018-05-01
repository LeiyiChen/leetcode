# letcode 简单等级刷题
### 1.两数之和
>给定一个整数数列，找出其中和为特定值的那两个数。
>你可以假设每个输入都只会有一种答案，同样的元素不能被重用。

**示例：**
>给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]

**我 6480ms**
```python
   for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                if nums[i] + nums[j] == target:
                    return [i,j]
```
**参考 48ms**
```python
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dic = dict()
        for index,value in enumerate(nums):
            sub = target - value
            if sub in dic:#只对字典的键进行查找
                return [dic[sub],index]
            else:
                dic[value] = index#将nums数组值和索引值存储在字典中，其中value为键，索引为值
```
### 7.颠倒整数
>给定一个范围为 32 位 int 的整数，将其颠倒。

例 1:

输入: 123
输出:  321
 

例 2:

输入: -123
输出: -321
 

例 3:

输入: 120
输出: 21
 

注意:

假设我们的环境只能处理 32 位 int 范围内的整数。根据这个假设，如果颠倒后的结果超过这个范围，则返回 0。

我：
```python
 l = list(str(x))
        n = len(l)
        for i in range(n-1,0,-1):
            if l[i] != '0':
                l = l[:i+1]
                break
        if l[0] =='-':
                    l = l[:0:-1]
                    l.insert(0,'-') 
        else:
            l = l[::-1]
        return int(''.join(l))
```
other1:
```python
 x = int(str(x)[::-1]) if x >0 else -int(str(-x)[::-1])
return x if x < 2147483648 and x >-2147483648 else 0
```
other2:
```python
        sign = x < 0 and -1 or 1
        x = abs(x)
        ans = 0
        while x:
            ans = ans * 10 + x % 10
            x /= 10
        return sign * ans if ans <= 0x7fffffff else 0
```

>总结：
1.python中字符串就可以完成分片操作，不需要再转换为list。用一个新的数据对象时要考虑清楚为什么用；
2.正数和负数只相差一个符号！如例2中设置一个符号位。1与任何数与都为任何数，0与任何数与都为0。与1 或是为了将0变为1.
3.对于32位的限制，去掉一个符号位，还剩31位。所以范围是正负2的31次方之间，可以像例2一样表示0x7ffffff.

### 9.回文数
判断一个整数是否是回文数。不能使用辅助空间。
我(这里要表扬一下，跟参考的写得差不多)
```python
        m = x
        s = 0
        while m!=0:
            s = s*10+(m%10)
            m = m//10
        return s == x#参考里用这一句替代了我以下的if语句
        ```
        if s == x:
            return True
        else:
            return False
        ```
```
缺点就是速度慢。
### 13.罗马数字转整数
>给定一个罗马数字，将其转换成整数。

返回的结果要求在 1 到 3999 的范围内。

我
```python
        d = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        result = 0
        i = 0
        while i < len(s)-1:
            if d[s[i]] >= d[s[i+1]]:
                result+= d[s[i]]
                i+=1
            else:
                temp = d[s[i+1]]-d[s[i]]
                result+=temp
                i+=2
        if i == len(s)-1:
            result+=d[s[i]]
        return result if 1 <= result <= 3999 else False    
```
别人
```python
        d = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        result = 0
        for i in range(len(s)-1):
            if d[s[i]] < d[s[i+1]]:
                result-= d[s[i]]
            else:
                result+=d[s[i]]
        result+=d[s[len(s)-1]]
        return result if 1 < result < 3999 else False
```
相关知识点在word文档中。
总结：思想要灵活，只知道加，不知道换个角度思考，不能加我可以减呐！
### 14.最长公共前缀
>编写一个函数来查找字符串数组中最长的公共前缀字符串。

解题分析：最长公共前缀指的是字符串数组中所有公共最长的前缀。
如果是空串的话，那么说明前缀就是“”
如果都是以“ ”开头的，那么就是“ ”
然后最长的前缀不会超过最短的字符串，那么可以遍历最短的字符串的长度，依次比较。
第一步：找出长度最短的字符串；
第二步：依次与长度最短的字符串比较。
我
```python
if strs == []:
            return ''
        common = []
        flag = 0
        def minimum(s):
            m = list(map(len,s))
            return s[m.index(min(m))]
        temp = minimum(strs)
        if temp == '':
            return ''
        a = temp[0]
        for i in strs:
            if i[0] != a:
                return ''
        for i in range(len(temp)):
            for j in strs:
                if temp[i]!=j[i]:
                    flag = 1
                    break
            if flag == 0:
                common.append(temp[i])
        return ''.join(common)
```
别人
```python
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]
        minl = min([len(x) for x in strs])
        end = 0
        while end < minl:
            for i in range(1,len(strs)):
                if strs[i][end]!= strs[i-1][end]:
                    return strs[0][:end]
            end += 1
        return strs[0][:end]
```
大神
```python
        res = ""
        if len(strs) == 0:
            return ""
        for each in zip(*strs):#zip()函数用于将可迭代对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            if len(set(each)) == 1:#利用集合创建一个无序不重复元素集
                res += each[0]
            else:
                return res
        return res
```
总结：1.不需要求出最短字符串是哪个，只需要求出最短字符串的长度即可。
2.字符串组的几种情况的判断应更清晰明了。
1）字符串组为空[]
2）公共字符串为空['','']
3)正常情况
### 20.有效的括号
>给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
括号必须以正确的顺序关闭，"()" 和 "()[]{}" 是有效的但是 "(]" 和 "([)]" 不是。

我(我的特点是写代码之前有个大致的思路就开始写，然后边写边改，导致无数个if，else逻辑不清。)
```python
        d = {'(':')','{':'}','[':']'}
        l = ''
        if len(s) == 1 or len(s) == 0:
            return False
        for  i in range(len(s)):
            print(s[i])
            if s[i] in d:
                l+=d[s[i]]
                print(l)
                if i ==len(s)-1:
                    return False
            elif s[i] == l[len(l)-1:]:
                l = l[:len(l)-1]
                print(l)
                if i == len(s)-1:
                    if l == '':
                        return True  
                    else:
                        return False
            else:
                return False   
```
别人一目了然
```python
        if len(s) % 2 == 1 or len(s) == 0:#这一步真的机智，我只会想到输入一个的状况。。。
            return False
        
        d = {'{': '}', '[': ']', '(': ')'}
        stack = []
        for i in s:
            # in stack
            if i in d:
                stack.append(i)
            else:
                if not stack or d[stack.pop()] != i:
                    return False
        ```
        else:
            if stack:
                return False
            
        return True
        ```
        return stack ==[]#以上四条语句可以用这条语句替代
```
最强王者
```python
        a = {')':'(', ']':'[', '}':'{'}
        l = [None]#设置None是为了排除空值的情况！
        for i in s:
            if i in a and a[i] == l[-1]:
                l.pop()
            else:
                l.append(i)
        return len(l)==1#用来排除空值的情况
```
总结：
1.拿到一个题首先要考虑有几种情况：
1)空和奇数个直接排除
2)剩下偶数个，若第一个就是右括号直接排除，不是右括号再入栈。
2.画个流程图辅助。
3.for else的用法。for循环完后执行else语句块，若for中有break，则跳过else。
4.l[-1]为l的最后一个元素。
### 21.合并两个有序链表
>合并两个已排序的链表，并将其作为一个新列表返回。新列表应该通过拼接前两个列表的节点来完成。 

示例
>输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

这题我写不出来，参考别人的写了好几遍依旧不流畅。所以这题应经常刷。
```python
 dummy = ListNode(0)
        s = dummy
        while l1 and l2:
            if l1.val > l2.val:
                s.next = l2
                l2 = l2.next
            else:
                s.next = l1
                l1 = l1.next
        if l1:
            while l1:
                s.next = l1
        if l2:
            while l2:
                s.next = l2
        return dummy.next
```
总结：这题关于链表的操作不熟练，从中看出我对类的理解还不透彻，对数据结构的理解也不透彻。
###26 从有序数组中删除重复项
>描述：
给定一个有序数组，你需要原地删除其中的重复内容，使每个元素只出现一次,并返回新的长度。
不要另外定义一个数组，您必须通过用 O(1) 额外内存原地修改输入的数组来做到这一点。

示例
>给定数组: nums = [1,1,2],
你的函数应该返回新长度 2, 并且原数组nums的前两个元素必须是1和2
不需要理会新的数组长度后面的元素

我,这是我第一个提交一次就通过的代码。
```python
        i = 0
        l = len(nums)
        if l:
            while i < l-1:
                if nums[i] == nums[i+1]:
                    nums.pop(i+1)
                    l-=1
                    j = i+1
                    while j < l:
                        if nums[i] != nums[j]:
                            break
                        else:
                            nums.pop(j)
                            l-=1
                i+=1
        return len(nums)
```
开心得不想看别人的代码了。
### 27.移除元素
>给定一个数组和一个值，在这个数组中原地移除指定值和返回移除后新的数组长度。
不要为其他数组分配额外空间，你必须使用 O(1) 的额外内存原地修改这个输入数组。
元素的顺序可以改变。超过返回的新的数组长度以外的数据无论是什么都没关系。

示例
>给定 nums = [3,2,2,3]，val = 3，
你的函数应该返回 长度 = 2，数组的前两个元素是 2。

我:不得了，这题十分钟就做好，零bug！！超越昨天一次性通过所有测试！！！
```python
        l = len(nums)
        if l == 0:
            return 0
        i = 0
        while i < l:
            if nums[i] == val:
                nums.pop(i)
                l-=1
            else:
                i+=1
        return len(nums)
```
总结：这一题和上一题都使用了，list.pop()方法，数组长度一直在发生变化。这个时候想要遍历数组的话，使用while比较合适，用for循环的话，遍历的索引值不方便控制其根据list长度变化而跟着变化。
比如：
```python
l = [1,2,3,4]
for i in l:
    print(i)
    l.pop(1)
    print(l)
```
以上代码块的运行结果为：
1#此时i访问的是l的第一个元素
2
[1, 3, 4]
3#此时i访问的是l的第二个元素
3
[1, 4]#下一步i应该访问l的第三个元素，但因为此时l只剩余两个元素，故运行结束

### 28实现strStr()
>实现 strStr()。
返回蕴含在 haystack 中的 needle 的第一个字符的索引，如果 needle 不是 haystack 的一部分则返回 -1 。

示例
>输入: haystack = "hello", needle = "ll"
输出: 2

>输入: haystack = "aaaaa", needle = "bba"
输出: -1

我：一次过。第一次感受到占了Python刷题的大便宜了，但是感觉不妥，还是应该正正经经从逻辑上写代码，不过三行代码是真的爽哈哈哈哈哈哈。
```python
        if needle  not in haystack:
            return -1
        return haystack.find(needle)
```
别人:虽然还是用了切片，不过这个方法很直接粗暴。效率更高
```python
    l = len(needle)
    for i in range(len(haystack)-l+1):
        if haystack[i:i+l] == needle:
            return i
    return -1
```
### 35.搜索插入位置
>给定一个排序数组和一个目标值，如果在数组中找到目标值则返回索引。如果没有，返回到它将会被按顺序插入的位置。
你可以假设在数组中无重复元素。

示例
>输入: [1,3,5,6], 5
输出: 2

>输入: [1,3,5,6], 2
输出: 1

>输入: [1,3,5,6], 7
输出: 4

>输入: [1,3,5,6], 0
输出: 0

我
```python
        if target in nums:
            return nums.index(target)
        else:
            if target < nums[0]:
                return 0
            elif target > nums[-1]:
                return len(nums)
            else:
                for i in range(len(nums)-1):
                    if nums[i] < target < nums[i+1]:
                        return i+1
```
总结：查找方式可以采用二分查找提高效率。
别人
```python
        lo = 0
        hi = len(nums)
        while lo < hi:
            mid = lo + (hi - lo) / 2
            if nums[mid] > target:
                hi = mid
            elif nums[mid] < target:
                lo = mid + 1
            else:
                return mid
        return lo
```
### 38数数并说
这题我不知道在说什么，略过。


### 53.最大子序和
描述
>给定一个序列（至少含有 1 个数），从该序列中寻找一个连续的子序列，使得子序列的和最大。
例如，给定序列 [-2,1,-3,4,-1,2,1,-5,4]，
连续子序列 [4,-1,2,1] 的和最大，为 6。

我 v1.0
```python
class Solution:
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = len(nums)
        i = 0
        result = nums[0]
        while i < l:
            sums = []
            temp = 0
            for j in range(i, l):
                temp+=nums[j]
                sums.append(temp)
            if result < max(sums):
                result = max(sums)
            i+=1
        return result
```
测试结果如下：
![](https://i.loli.net/2018/04/12/5aced62975043.png)
本地运行时间为14.7s，说明我的方法太粗暴了。应该寻找更好的算法。
![屏幕快照 2018-04-12 上午11.52.43.png](https://i.loli.net/2018/04/12/5aced81b2033c.png)
我 优化后v1.1。优化方案，去掉sums数组，节省空间。但时间复杂度仍然不变。
```python
        l = len(nums)
        i = 0
        result = nums[0]
        while i < l:
            temp = 0
            for j in range(i, l):
                temp+=nums[j]
                if result < temp:
                    result = temp
            i+=1
        return result
```
仍然只通过200/202测试用例，仍然超出时间限制。但本地运行时间为8.3s。有进步。
别人，分治法。时间复杂度O(NlogN)
将输入的序列分成两部分，这个时候有三种情况。
1）最大子序列在左半部分
2）最大子序列在右半部分
3）最大子序列跨越左右部分。
前两种情况通过递归求解，第三种情况可以通过。
分治法代码大概如下，emmm。。。目前还没有完全理解。分析可参考https://blog.csdn.net/samjustin1/article/details/52043173
```python

def maxC2(ls,low,upp):  
    #"divide and conquer"  
    if ls is None: return 0  
    elif low==upp: return ls[low]  
      
    mid=(low+upp)/2 #notice: in the higher version python, “/” would get the real value  
    lmax,rmax,tmp,i=0,0,0,mid  
    while i>=low:  
        tmp+=ls[i]  
        if tmp>lmax:  
            lmax=tmp  
        i-=1  
    tmp=0  
    for k in range(mid+1,upp):  
        tmp+=ls[k]  
        if tmp>rmax:  
            rmax=tmp  
    return max3(rmax+lmax,maxC2(ls,low,mid),maxC2(ls,mid+1,upp))  
  
def max3(x,y,z):  
    if x>=y and x>=z:  
        return x  
    return max3(y,z,x)  
```
动态规划算法，时间复杂度为O(n)。
分析：寻找最优子结构。
```python
        l = len(nums)
        i = 0
        sum = 0
        MaxSum = nums[0]
        while i < l:
            sum+=nums[i]
            if sum > MaxSum:
                    MaxSum = sum
            if sum < 0:
                sum = 0
            i+=1
        return MaxSum
```
Oh！My god！！！ ！！！！！！！！运行只花了0.2s！！！！！！！！！！！！！！！这也太强了吧！！
![动态规划](https://i.loli.net/2018/04/13/5ad0267adbbc8.png)
优化后,运行时间0.1s.
```python
        sum = 0
        MaxSum = nums[0]
        for i in range(len(nums)):
            sum += nums[i]
            if sum > MaxSum:
                MaxSum = sum
            if sum < 0:
                sum = 0
        return MaxSum
```
其中
`sum += nums[i]`必须紧挨。
```if sum > MaxSum:
                MaxSum = sum
```
### 58.最后一个单词的长度
>描述
给定一个字符串， 包含大小写字母、空格 ' '，请返回其最后一个单词的长度。
如果不存在最后一个单词，请返回 0 。
注意事项：一个单词的界定是，由字母组成，但不包含任何的空格。

>案例
输入: "Hello World"
输出: 5

我
```python
class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        l = len(s)
        if l == 0:
            return 0
        n = l-1
        while n >= 0:
            if s[n] == ' ':
                n-=1
            else:
                for i in range(n-1,-1,-1):
                    if s[i] == ' ':
                        return n - i
                return n+1
        return 0
```
别人：使用了split()方法、strip()方法。或者如下
```python
        str = ''
        count = 0
        for i in s[::-1]:
            if str != '' and i==' ':
                return count
            if i != ' ':
                count = count+ 1
                str = str + i
        return count
```
### 66.加一
>描述：给定一个非负整数组成的非空数组，在该数的基础上加一，返回一个新的数组。
最高位数字存放在数组的首位， 数组中每个元素只存储一个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

示例1
>输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。

示例2
>输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。

我
```python
        l = digits[::]
        i = len(digits)-1
        while i >= 0:
            if l[i]+1 <=9:
                l[i]+=1
                return l
            else:           
                if i == 0:
                    l[i] = (l[i]+1)%10
                    l.insert(0,1)
                    return l        
                elif l[i]+1 > 9:
                    l[i]=(l[i]+1)%10
                    if l[i-1]+1 < 10:
                        l[i-1] = l[i-1]+1
                        return l
                    else:
                        i-=1
```
先不说我的代码，先看看别人的，如此干净利落！
```python
        if len(digits) == 0:
            digits = [1]
        elif digits[-1] == 9:
            digits = self.plusOne(digits[:-1])
            digits.extend([0])
        else:
            digits[-1] += 1
        return digits
```
我的思想太粗暴了，题目中说最后一个数要加1.我就老老实实把最后一个元素加一来判断。分两种情况：
1）加1以后小于9。
2）加1以后大于9。但是这个情况又涉及两种情况：a.digits只有一个元素，所以在index=0插入1。b.相加以后产生进位，若将进位加到前面一位数仍产生进位怎么处理。
emmm...我jio得我对步骤的描述并不科学。

大神的就很简洁。题目中也有说digits里的数是1-9，所以只要判断最后一位数是不是9就好。如果是9的话，那个位置就更改为0.而不用向我的一样，再进行求余运算。emmm。。。我把问题的范围扩大了，我的程序可以求解1-18之间的数。这里就两种情况了：最后一位不是9；最后一位是9，那么相加以后会产生进位，就得再接着判断前一位是不是9.这里可以采用循环（像我一样）或者像大神一样（采用递归）。递归的话可以尝试单步调试，有助于理解过程。
### 67.二进制求和
描述
>给定两个二进制字符串，返回他们的和（用二进制表示）。
输入为非空字符串且只包含数字 1 和 0。

示例1
>输入: a = "11", b = "1"
输出: "100"

示例2
>输入: a = "1010", b = "1011"
输出: "10101"

我
```python
        buff = 0
        s = ''
        flag = 1
        flag1 = 1
        l = min(len(a), len(b))
        if len(a) > len(b):
            temp = a[len(a) - l:]
            min_string = b
            other = a[:len(a) - l]
        elif len(a) < len(b):
            temp = b[len(b) - l:]
            min_string = a
            other = b[:len(b) - l]
        else:
            flag1 = 0
            temp = a
            min_string = b
        for i in range(l - 1, -1, -1):
            if buff == 0:
                if temp[i] == min_string[i] == '0':
                    s += '0'
                    buff = 0
                elif temp[i] == min_string[i] == '1':
                    s += '0'
                    buff = 1
                else:
                    s += '1'
                    buff = 0
            else:
                if temp[i] == min_string[i] == '0':
                    s += '1'
                    buff = 0
                elif temp[i] == min_string[i] == '1':
                    s += '1'
                    buff = 1
                else:
                    s += '0'
                    buff = 1
        if not flag1:
            if buff == 1:
                s += '1'
        elif not buff:
            s += other[::-1]
        else:
            for j in range(len(other) - 1, -1, -1):
                if other[j] == '0':
                    s += '1'
                    if j == 1:
                        s+=other[:1]
                    else:
                        other = other[:j]
                        s += other[::-1]
                    flag = 0
                    break
                else:
                    s += '0'
                    if j == 0:
                        s += '1'
        return s[::-1]
```
代码调了两个多小时，越写越复杂，终于通过。错误主要出现在切片反转。结果一看别人的就两行代码，扎心了！
两行代码
```python
        a, b = int('0b' + a, 2), int('0b' + b, 2)
        return bin(a + b)[2:]
```
emmm...我该说什么呢。哈哈哈让我们看看一行代码的吧！！
```python
        return bin(int(a,2)+int(b,2))[2:]
```
还有另外长一点但比我短的代码，是将a,b反转过来，这样列表切片方便。
int()函数用于将一个字符串或数字转换为整型。语法：
class int(x, base)
x--字符串或数字
base--进制数，默认十进制。
bin()函数返回一个整型int或者长整数long int的二进制表示。
![](https://i.loli.net/2018/04/21/5adb334b9b1b4.png)
bin()运算返回的是二进制。所以前两位是二进制的标志，需要[2:]去除。
### 69.x的平方根
描述
>实现 int sqrt(int x) 函数。
计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

示例1
>输入: 4
输出: 2

示例2
>输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。

我第一想法简单粗暴但超出时间限制。
```python
        if x == 0:
            return 0
        elif x < 0:
            return 'error'
        else:
            i = 1
            while True:
                if (i+1)**2 > x >= i**2:
                    return i
                i+=1
```
改进一下,因为结果肯定是有序的从1，2，3.。。查找，所以使用二分查找。
```python
        if x == 1:
            return 1
        l = 1
        h = x//2
        while h > l:
            m = (l+h)//2
            if m**2 > x:
                h = m-1
            elif m**2 < x:
                l = m+1
            else:
                return m
        if h == l:
            if (h-1)**2 < x < h**2:
                return h-1
            else:
                return h
        elif h < l:
            return h
```
这里比二分查找麻烦的是需要考虑，当结果不在查找的序列是按下取整。这下终于通过了，但是运行时间却是吊车尾。看看别人的，也是用的二分查找，不过是在我的第一版基础上改了一下，不像我这么笨笨地机械套用二分查找。
```python
        if x == 0:
            return 0
        
        l = 1
        r = x
        
        while l <= x:
            res = (l + r) // 2
            s = res**2
            
            if s <= x < (res + 1)**2:
                return res
            if s < x:
                l = res
            if s > x:
                r = res
```
这个问题关键在于结果的下取整。所以我的代码在判断x与m平方的时候可以进行优化。(m-1)**2 <= x < m**2.
### 70.爬楼梯
描述
>假设你正在爬楼梯。需要 n 步你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。

示例1
>输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 步 + 1 步
2.  2 步

示例2
>输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 步 + 1 步 + 1 步
2.  1 步 + 2 步
3.  2 步 + 1 步

我第一反应是用递归实现，然而不出所料超出时间限制。
```python
        if n == 1:
            return 1
        elif n == 2:
            return 2
        else:
            s1 = self.climbStairs(n-1)
            s2 = self.climbStairs(n-2)
            return s1+s2 
```
所以。。。。再想一想。既然递归不行就递推试试。emmm...网上说这个问题的结果和Fibonacci数列一样。果然是这样。。。尽管可以转换为求解fibonacci，但是还是应该好好思考一下怎么解决原问题。毕竟可能会联想不到fibonacci。
```python
        nums = [0,1,2]
        if n == 1:
            return nums[1]
        elif n == 2:
            return nums[2]
        else:
            for i in range(3,n+1):
                nums.append(nums[i-1] + nums[i-2])
        return nums[n]
```
上面这个版本通过了，但是只战胜了25%的人，寻求更优的算法吧。
这个很机智啊，我的列表初始化可以优化成如下：
```python
        condition = [0] * (n + 1)
        condition[0] = 1
        condition[1] = 1
        for i in range(2, n+1):
            condition[i] = condition[i-1] + condition[i-2]
        return condition[n]
```
关于循环和递归：
Loops may achieve a performance gain for your program. Recursion may achieve a performance gain for your programmer. Choose which is more important in your situation!
如果使用循环，程序的性能会更高。如果使用递归，程序更容易理解。
### 83.删除排序链表中的重复元素
描述
>给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

示例1
>输入: 1->1->2
输出: 1->2

示例2
>输入: 1->1->2->3->3
输出: 1->2->3

我
```python
        if not head:
            return None
        p = head
        if p.next == None:
            return head
        try:
            while p.next:
                i = p.val
                if p.next.val == i:
                    p.next = p.next.next
                    while p.next:
                        if p.next.val != i:
                            break
                        p.next = p.next.next
                p = p.next
        except AttributeError:
            return head
        else:
            return head
```
写了两三个小时吧，一直改改改。搞得最后逻辑都不是很清楚了，所以引入了异常处理。。。。。然后发现leetcode中输入示例中[1,2,3]头节点是指1.所以返回的是head而不是head.next。
别人的if...else判断一目了然，逻辑很清晰。
```python
class Solution:
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None:
            return None
        cur = head
        while cur.next:

            if cur.val == cur.next.val:
                if cur.next.next is None:
                    cur.next = None
                else:
                    temp = cur.next.next
                    cur.next = temp
            else:
                cur = cur.next

        return head
```
### 88.合并两个有序数组
描述
>给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。
初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。

示例
>输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3
输出: [1,2,2,3,5,6]

我
```python
        if n == 0:
            return nums1
        elif n == 0:
            return nums2
        if nums2[-1] <= nums1[0]:
            return nums2 + nums1
        for j in range(len(nums2)):
            if nums2[j] >= nums1[-1]:
                nums1.extend(nums2[j:])
                break
            else:
                l = 0
                h = len(nums1)
                while l < h:
                    m = (l+h)//2
                    if nums2[j] <= nums1[m]:
                        if nums1[m-1] < nums2[j]:
                            nums1.insert(m,nums2[j])
                            break
                        else:
                            h = m-1
                    else: 
                        if nums1[m+1] > nums2[j]:
                            nums1.insert(m+1, nums2[j])
                            break
                        else:
                            l = m+1
        return nums1
```
这么多代码，理所当然超时了呀。。。我的代码丑哭了。并且我并没有假设nums1后面用0来表示nums2占用的空间。思路是，依次遍历nums2中的元素，作为target，然后在nums1中使用二分查找，并插入target。
利用归并排序。
别人：
```python
        end = m + n -1
        m-=1
        n-=1
        while end >= 0 and m >= 0 and n >= 0:
            if nums1[m] >= nums2[n]:
                nums1[end] = nums1[m]
                m-=1
            else:
                nums1[end] = nums2[n]
                n-=1
            end-=1

        while n >= 0:
            nums1[end] = nums2[n]
            n-=1
            end-=1
```
###100.相同的树
描述：
>给定两个二叉树，编写一个函数来检验它们是否相同。
如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

[![示例](https://i.loli.net/2018/04/29/5ae564bc6a702.png)](https://i.loli.net/2018/04/29/5ae564bc6a702.png)

[![](https://i.loli.net/2018/04/29/5ae564dd7cff2.png)](https://i.loli.net/2018/04/29/5ae564dd7cff2.png)

树要遍历每个节点，所以无论怎么写都离不开递归。
```python
        if p == None or q == None:
            return p == q
        elif p.val == q.val:
            return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)
        else:
            return False
```
别人的。。。
```python
        if p == None or q == None:
            return p == q
        return p.val == q.val and self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)
```
### 101.对称二叉树
描述
[![](https://i.loli.net/2018/04/29/5ae5635849d4f.png)](https://i.loli.net/2018/04/29/5ae5635849d4f.png)
我。。。看了别人的跌跌爬爬写出来的。。。
```python
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def f(p, q):
            if p == None:
                return q == None
            if q == None:
                return p == None
            if p.val == q.val:
                return f(p.left, q.right) and f(p.right, q.left)
            if p.val != q.val:
                return False
        if root == None:
            return True
        return f(root.left, root.right)
```
关于二叉树的总结：对于树的操作，是必定要使用递归的方法的。对于求解树的任何问题，都先使用分治法，将树拆分为左子树和右子树（这里主要针对二叉树来说明），如果左子树和右子树都符合条件，则这颗树都满足条件。而左子树和右子树再拆分成左子树和右子树，进行递归。这里需要注意的是递归结束的条件，以及递归过程。
### 104.二叉树的最大深度
描述
[![](https://i.loli.net/2018/04/30/5ae701053c2ad.png)](https://i.loli.net/2018/04/30/5ae701053c2ad.png)
我
```python
        def depth(p):
            if p == None:
                return 1
            else:
                return 1 + max(depth(p.right), depth(q.left))
        
        if root == None:
            return 0
        else:
            return max(depth(root.right), depth(root.left))
```
ok！二十分钟以内一次性通过搞定！！不妄研究了两天的树！！
别人依然两行搞定
```python
        if root == None:
            return 0
        else:
            return 1 + max(maxDepth(root.right), maxDepth(root.left))
```
### 107.二叉树的层次遍历 II
描述
[![](https://i.loli.net/2018/04/30/5ae708055ce4c.png)](https://i.loli.net/2018/04/30/5ae708055ce4c.png)
我又研究了一下树的四种遍历，不然之前学的全忘光了。温故而知新！
```python
    if root is None:
        return []
    p = [root]
    results = []
    current_level_num = 1
    next_level_num = 0
    d = []
    while p:
        current = p.pop(0)
        d.append(current.val)
        current_level_num -= 1
        if current.left != None:
            p.append(current.left)
            next_level_num += 1
        if current.right != None:
            p.append(current.right)
            next_level_num += 1
        if current_level_num == 0:
            current_level_num = next_level_num
            next_level_num = 0
            results.append(d)
            d = []
    return results[::-1]
```





